// ============================================================================
// poseEstimation3D — thin ROS 1 client for the SAM3DBody-cpp net server.
//
// Subscribes to an RGB image topic, JPEG-encodes each frame, POSTs it to the
// `sam_3dbody_net --server` (via AmmClient, the minimal HTTP client vendored
// from AmmarServer), parses the SAM3D text response and broadcasts the
// skeleton as a ROS TF tree + a std_msgs::Float32MultiArray — the same way
// FORTH's mocapnet_rosnode does.  All pose estimation (YOLO + DINOv3 backbone +
// MHR + FK) runs REMOTELY on the GPU server, so this node only does JPEG encode
// and TF publishing: it gives an old ROS-1 robot a brand-new pose stack with no
// GPU and no heavy dependencies on the robot.
//
// Coordinate handling mirrors mocapnet_rosnode: every joint transform is
// parent-relative, and the whole skeleton hangs off a single camera frame whose
// orientation relative to tfRoot (the cameraRoll/Pitch/Yaw params) performs the
// camera-optical -> ROS (REP-103) remap.  Tune those three numbers if the
// skeleton appears rotated.
// ============================================================================

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/String.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "AmmClient.h"
}

// ── Configuration (filled from ROS params in main) ───────────────────────────
static std::string g_serverHost        = "127.0.0.1";
static int         g_serverPort        = 8080;
static int         g_timeoutS          = 10;
static int         g_jpegQuality       = 80;
static std::string g_nodeName          = "poseEstimation3D";
static std::string g_tfRoot            = "map";
static std::string g_cameraFrame       = "poseEstimation3DCamera";
static std::string g_personPrefix      = "p";          // -> p0/<joint>, p1/<joint>
static int         g_publishCameraTF   = 1;
static int         g_useSimple3DPointTF = 0;            // 1 = KP3D points (no --jlocal)
static double      g_camX = 0.0, g_camY = 0.0, g_camZ = 0.0;
// Default = standard camera-optical -> ROS REP-103 orientation, in DEGREES.
static double      g_camRollDeg  = -90.0;
static double      g_camPitchDeg =   0.0;
static double      g_camYawDeg   = -90.0;

static AmmClient_Instance* g_client = 0;
static ros::Publisher      g_skelPub;   // Float32MultiArray (numeric)
static ros::Publisher      g_rawPub;    // String (lossless SAM3D text)

static inline double deg2rad(double d) { return d * M_PI / 180.0; }

// ── TF broadcast (same shape as mocapnet_rosnode::postPoseTransform) ─────────
static void postPoseTransform(const char* parent, const char* child,
                              float x, float y, float z,
                              float qx, float qy, float qz, float qw,
                              const ros::Time& stamp)
{
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped t;
    t.header.stamp    = stamp;
    t.header.frame_id = parent;
    t.child_frame_id  = child;
    t.transform.translation.x = x;
    t.transform.translation.y = y;
    t.transform.translation.z = z;
    t.transform.rotation.x = qx;
    t.transform.rotation.y = qy;
    t.transform.rotation.z = qz;
    t.transform.rotation.w = qw;
    br.sendTransform(t);
}

// Read a full HTTP response: loop AmmClient_Recv until the header terminator
// (\r\n\r\n) AND Content-Length body bytes are present (AmmarServer sends the
// header and body as separate segments).  Returns body pointer or 0.
static char* recvHttpResponse(AmmClient_Instance* inst,
                              std::vector<char>& buf, unsigned int* outLen)
{
    unsigned int total = 0;
    long contentLen = -1;
    while (total + 1 < buf.size())
    {
        unsigned int chunk = (unsigned int)(buf.size() - 1 - total);
        if (!AmmClient_Recv(inst, buf.data() + total, &chunk)) return 0;
        if (chunk == 0) break;                       // peer closed
        total += chunk;
        buf[total] = 0;                              // null-terminate for strstr
        unsigned int afterHdr = total;
        char* body = AmmClient_seekEndOfHeader(buf.data(), &afterHdr);
        if (!body) continue;
        if (contentLen < 0) {
            char* cl = strstr(buf.data(), "Content-Length:");
            if (cl) contentLen = atol(cl + 15);
        }
        if (contentLen < 0) { *outLen = afterHdr; return body; }
        if (afterHdr >= (unsigned int)contentLen) { *outLen = (unsigned int)contentLen; return body; }
    }
    return 0;
}

// ── Main per-frame callback ──────────────────────────────────────────────────
static void rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
    // 1. ROS image -> BGR cv::Mat
    cv::Mat bgr;
    try {
        bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(5.0, "[poseEstimation3D] cv_bridge: %s", e.what());
        return;
    }
    if (bgr.empty()) return;

    // 2. JPEG encode
    std::vector<uchar> jpeg;
    std::vector<int> params; params.push_back(cv::IMWRITE_JPEG_QUALITY); params.push_back(g_jpegQuality);
    if (!cv::imencode(".jpg", bgr, jpeg, params)) {
        ROS_WARN_THROTTLE(5.0, "[poseEstimation3D] jpeg encode failed");
        return;
    }

    // 3. POST to the server, read the SAM3D text response
    if (!AmmClient_SendFile(g_client, "/infer", "uploadedfile", "f.jpg", "image/jpeg",
                            (const char*)jpeg.data(), (unsigned int)jpeg.size(), /*keepAlive=*/1)) {
        ROS_WARN_THROTTLE(5.0, "[poseEstimation3D] send to %s:%d failed",
                          g_serverHost.c_str(), g_serverPort);
        return;
    }
    static std::vector<char> rbuf(1 << 20);
    unsigned int bodyLen = 0;
    char* body = recvHttpResponse(g_client, rbuf, &bodyLen);
    if (!body) { ROS_WARN_THROTTLE(5.0, "[poseEstimation3D] no response body"); return; }

    const ros::Time stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;

    // 4. Camera frame: one transform that reorients the whole tree into ROS.
    if (g_publishCameraTF) {
        tf2::Quaternion cq;
        cq.setRPY(deg2rad(g_camRollDeg), deg2rad(g_camPitchDeg), deg2rad(g_camYawDeg));
        postPoseTransform(g_tfRoot.c_str(), g_cameraFrame.c_str(),
                          (float)g_camX, (float)g_camY, (float)g_camZ,
                          (float)cq.x(), (float)cq.y(), (float)cq.z(), (float)cq.w(), stamp);
    }

    // 5. Parse SAM3D text -> TF tree + Float32MultiArray.
    std::string s(body, bodyLen);
    std_msgs::Float32MultiArray arr;     // [ id, njoints, (qx qy qz qw tx ty tz)*njoints ] per person
    std::vector<float>& d = arr.data;
    long njFixupIdx = -1;                // index of the current person's njoints slot
    int  njCount    = 0;
    int  personIdx  = -1, personId = -1;

    size_t pos = 0;
    while (pos < s.size())
    {
        size_t eol = s.find('\n', pos);
        if (eol == std::string::npos) eol = s.size();
        const std::string line = s.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.empty()) continue;

        if (line.compare(0, 2, "P ") == 0) {
            if (njFixupIdx >= 0) d[njFixupIdx] = (float)njCount;   // close previous person
            if (sscanf(line.c_str(), "P %d id=%d", &personIdx, &personId) == 2) {
                d.push_back((float)personId);
                njFixupIdx = (long)d.size(); d.push_back(0.0f);    // njoints placeholder
                njCount = 0;
            }
        }
        else if (!g_useSimple3DPointTF && line.compare(0, 7, "JLOCAL ") == 0 && personIdx >= 0) {
            char joint[80], parent[80]; float q[4], t[3];
            if (sscanf(line.c_str(), "JLOCAL %79s %79s %f %f %f %f %f %f %f",
                       joint, parent, &q[0], &q[1], &q[2], &q[3], &t[0], &t[1], &t[2]) == 9) {
                char childFrame[200], parentFrame[200];
                snprintf(childFrame, sizeof(childFrame), "%s%d/%s",
                         g_personPrefix.c_str(), personIdx, joint);
                if (strcmp(parent, "camera") == 0)
                    snprintf(parentFrame, sizeof(parentFrame), "%s", g_cameraFrame.c_str());
                else
                    snprintf(parentFrame, sizeof(parentFrame), "%s%d/%s",
                             g_personPrefix.c_str(), personIdx, parent);
                postPoseTransform(parentFrame, childFrame, t[0], t[1], t[2],
                                  q[0], q[1], q[2], q[3], stamp);
                d.push_back(q[0]); d.push_back(q[1]); d.push_back(q[2]); d.push_back(q[3]);
                d.push_back(t[0]); d.push_back(t[1]); d.push_back(t[2]);
                ++njCount;
            }
        }
        else if (g_useSimple3DPointTF && line.compare(0, 5, "KP3D ") == 0 && personIdx >= 0) {
            // "KP3D <n> x,y,z x,y,z ..." — positions only, identity rotation.
            const char* c = line.c_str() + 5;
            int n = 0, consumed = 0;
            if (sscanf(c, "%d%n", &n, &consumed) == 1) {
                c += consumed;
                float x, y, z; int k = 0;
                while (k < n && sscanf(c, " %f,%f,%f%n", &x, &y, &z, &consumed) == 3) {
                    c += consumed;
                    char child[200];
                    snprintf(child, sizeof(child), "%s%d/kp%d", g_personPrefix.c_str(), personIdx, k);
                    postPoseTransform(g_cameraFrame.c_str(), child, x, y, z, 0, 0, 0, 1, stamp);
                    d.push_back((float)personId); d.push_back((float)k);
                    d.push_back(x); d.push_back(y); d.push_back(z);
                    ++k;
                }
            }
        }
    }
    if (njFixupIdx >= 0) d[njFixupIdx] = (float)njCount;

    // 6. Publish.  TF is the primary structured output; these mirror mocapnet's
    //    Float32MultiArray, plus a lossless raw-text topic for easy parsing.
    if (!d.empty()) {
        std_msgs::MultiArrayDimension dim; dim.label = "data"; dim.size = d.size(); dim.stride = d.size();
        arr.layout.dim.push_back(dim);
        g_skelPub.publish(arr);
    }
    std_msgs::String raw; raw.data = s; g_rawPub.publish(raw);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "poseEstimation3D");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string fromRGBTopic;
    pnh.param("name",               g_nodeName,        std::string("poseEstimation3D"));
    pnh.param("serverHost",         g_serverHost,      std::string("127.0.0.1"));
    pnh.param("serverPort",         g_serverPort,      8080);
    pnh.param("socketTimeout",      g_timeoutS,        10);
    pnh.param("jpegQuality",        g_jpegQuality,     80);
    pnh.param("fromRGBTopic",       fromRGBTopic,      std::string("/camera/rgb/image_raw"));
    pnh.param("tfRoot",             g_tfRoot,          std::string("map"));
    pnh.param("cameraFrame",        g_cameraFrame,     std::string("poseEstimation3DCamera"));
    pnh.param("personFramePrefix",  g_personPrefix,    std::string("p"));
    pnh.param("publishCameraTF",    g_publishCameraTF, 1);
    pnh.param("useSimple3DPointTF", g_useSimple3DPointTF, 0);
    pnh.param("cameraXPosition",    g_camX,            0.0);
    pnh.param("cameraYPosition",    g_camY,            0.0);
    pnh.param("cameraZPosition",    g_camZ,            0.0);
    pnh.param("cameraRoll",         g_camRollDeg,      -90.0);
    pnh.param("cameraPitch",        g_camPitchDeg,       0.0);
    pnh.param("cameraYaw",          g_camYawDeg,       -90.0);

    g_client = AmmClient_Initialize(g_serverHost.c_str(), g_serverPort, g_timeoutS);
    if (!g_client) {
        ROS_FATAL("[poseEstimation3D] could not init AmmClient for %s:%d",
                  g_serverHost.c_str(), g_serverPort);
        return 1;
    }

    g_skelPub = nh.advertise<std_msgs::Float32MultiArray>(g_nodeName + "/skeleton", 10);
    g_rawPub  = nh.advertise<std_msgs::String>(g_nodeName + "/raw", 10);

    // queue size 1 = drop-latest: while we block on the synchronous HTTP round
    // trip, intermediate camera frames are dropped so we stay real-time.
    ros::Subscriber sub = nh.subscribe(fromRGBTopic, 1, rgbCallback);

    ROS_INFO("[poseEstimation3D] server=%s:%d  image=%s  mode=%s  tfRoot=%s camera=%s",
             g_serverHost.c_str(), g_serverPort, fromRGBTopic.c_str(),
             g_useSimple3DPointTF ? "KP3D-points" : "JLOCAL-tree (needs --jlocal)",
             g_tfRoot.c_str(), g_cameraFrame.c_str());

    ros::spin();

    AmmClient_Close(g_client);
    return 0;
}
