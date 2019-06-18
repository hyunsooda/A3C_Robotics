#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <boost/thread/mutex.hpp>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <map>

using namespace cv;
using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
#define toRadian(degree)	((degree) * (M_PI / 180.))
#define toDegree(radian)	((radian) * (180. / M_PI))



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global variable
boost::mutex mutex[2];
nav_msgs::Odometry g_odom;
sensor_msgs::LaserScan g_scan;
vector<Vec3d> trajectory;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A template method to check 'nan'
template<typename T>
inline bool isnan(T value)
{
    return value != value;
}

bool checkAngle(int angle, int limit) {
    return (angle + limit/2) % 360 <= limit;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// callback function
void
odomMsgCallback(const nav_msgs::Odometry &msg)
{
    // receive a '/odom' message with the mutex
    mutex[0].lock(); {
        g_odom = msg;
    } mutex[0].unlock();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// callback function
void
scanMsgCallback(const sensor_msgs::LaserScan& msg)
{
    // receive a '/odom' message with the mutex
    mutex[1].lock(); {
        g_scan = msg;
    } mutex[1].unlock();
}

void resultMsgCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    trajectory.clear();
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
convertOdom2XYZRPY(nav_msgs::Odometry &odom, Vec3d &xyz, Vec3d &rpy)
{
    // 이동 저장
    xyz[0] = odom.pose.pose.position.x;
    xyz[1] = odom.pose.pose.position.y;
    xyz[2] = odom.pose.pose.position.z;

    // 회전 저장
    tf::Quaternion rotationQuat = tf::Quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w);
    tf::Matrix3x3(rotationQuat).getEulerYPR(rpy[2], rpy[1], rpy[0]);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
convertScan2XYZs(sensor_msgs::LaserScan& lrfScan, vector<Vec3d> &XYZs)
{
    int nRangeSize = (int)lrfScan.ranges.size();
    XYZs.clear();
    XYZs.resize(nRangeSize);

    for(int i=0; i<nRangeSize; i++) {
        double dRange = lrfScan.ranges[i];

        if(isnan(dRange)) {
            XYZs[i] = Vec3d(0., 0., 0.);
        } else {
            double dAngle = lrfScan.angle_min + i*lrfScan.angle_increment;
            XYZs[i] = Vec3d(dRange*cos(dAngle), dRange*sin(dAngle), 0.);
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
saveCurrentPosition(Vec3d &xyz, vector<Vec3d> &trajectory, double dMinDist)
{
    int nSize = (int) trajectory.size();

    if(nSize <= 0) {
        trajectory.push_back(xyz);
    } else {
        Vec3d diff = trajectory[nSize-1] - xyz;
        double len = sqrt(diff.dot(diff));

        if(len > dMinDist) {
            trajectory.push_back(xyz);
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
vector<Vec3d> transform(vector<Vec3d> laserScanXY, double x, double y, double theta)
{
    vector<Vec3d> ret;
    Vec3d newPt;
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);
    int nRangeSize = (int)laserScanXY.size();

    for(int i=0; i<nRangeSize; i++) {
        auto newX = laserScanXY[i][0] - x;
        auto newY = laserScanXY[i][1] - y;
        newPt[0] = cosTheta*newX + -1.*sinTheta*newY;
        newPt[1] = sinTheta*newX + cosTheta*newY;
        newPt[2];
        ret.push_back(newPt);
    }

    return ret;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
initGrid(Mat &display, int nImageSize)
{
    const int nImageHalfSize = nImageSize/2;
    const int nAxisSize = nImageSize/16;
    const Vec2i imageCenterCooord = Vec2i(nImageHalfSize, nImageHalfSize);
    display = Mat::zeros(nImageSize, nImageSize, CV_8UC3);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
drawTrajectory(Mat &display, vector<Vec3d> &trajectory, double dMaxDist)
{
    Vec2i imageHalfSize = Vec2i(display.cols/2, display.rows/2);

    int nSize = (int) trajectory.size();

    for(int i=1; i<nSize; i++) {
        int x = imageHalfSize[0] + cvRound((trajectory[i][0]/dMaxDist)*imageHalfSize[0]);
        int y = imageHalfSize[1] + cvRound((trajectory[i][1]/dMaxDist)*imageHalfSize[1]);

        if(x >= 0 && x < display.cols && y >= 0 && y < display.rows) {
            display.at<Vec3b>(y, x) = Vec3b(128, 128, 128);
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
drawCurrentPositionWithRotation(Mat &display, Vec3d &xyz, Vec3d &rpy, double dMaxDist)
{
    printf("_r = %.3lf, _p = %.3lf, _y = %.3lf\n", toDegree(rpy[0]), toDegree(rpy[1]), toDegree(rpy[2]));

    const int nHeadingSize = 20;
    Vec2i headingDir = Vec2i(nHeadingSize*cos(rpy[2]), nHeadingSize*sin(rpy[2]));
    Vec2i imageHalfSize = Vec2i(display.cols/2, display.rows/2);

    int x = imageHalfSize[0] + cvRound((xyz[0]/dMaxDist)*imageHalfSize[0]);
    int y = imageHalfSize[1] + cvRound((xyz[1]/dMaxDist)*imageHalfSize[1]);

    if(x >= 0 && x < display.cols && y >= 0 && y < display.rows) {
        circle(display, Point(display.cols/2, display.rows/2), nHeadingSize, CV_RGB(255, 255, 255), 1, CV_AA);
        line(display, Point(display.cols/2, display.rows/2), Point(display.cols/2 + nHeadingSize * sqrt(3) / 2, display.rows/2 + nHeadingSize / 2), CV_RGB(255, 255, 255), 1, CV_AA);
        line(display, Point(display.cols/2, display.rows/2), Point(display.cols/2 + nHeadingSize * sqrt(3) / 2, display.rows/2 - nHeadingSize / 2), CV_RGB(255, 255, 255), 1, CV_AA);
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A callback function. Executed eack time a new pose message arrives.
pair<int, double> drawLRFScan(Mat &display, vector<Vec3d> &laserScanXY, double dMaxDist)
{
    Vec2i imageHalfSize = Vec2i(display.cols/2, display.rows/2);
    int nRangeSize = (int)laserScanXY.size();
    vector<vector<int> > obstacles;
    vector<int> group;
    tuple<double, int, int> nearest = make_tuple(-1., 0, 0);
    const double isSame = 0.25, nearestMin = 1.;
    double prevX, prevY;
    int cnt = 0;
    bool isFront = false;

    //시작점부터 반대방향으로 장애물이 끝날때까지 한번 훑는다
    if(nRangeSize > 0) {
        prevX = laserScanXY[0][0];
        prevY = laserScanXY[0][1];
    }
    while(nRangeSize > 0) {
        double nowX = laserScanXY[--nRangeSize][0];
        double nowY = laserScanXY[nRangeSize][1];
        int x = imageHalfSize[0] + cvRound((nowX/dMaxDist)*imageHalfSize[0]);
        int y = imageHalfSize[1] + cvRound((nowY/dMaxDist)*imageHalfSize[1]);
        if(x >= 0 && x < display.cols && y >= 0 && y < display.rows) {
            if(isSame >= sqrt((prevX-nowX)*(prevX-nowX)+(prevY-nowY)*(prevY-nowY))) {
                if(checkAngle(nRangeSize, 60) && (get<0>(nearest) < 0 || get<0>(nearest) > sqrt(nowX*(-nowX)+(-nowY)*(-nowY))))
                    nearest = make_tuple(sqrt(nowX*nowX+nowY*nowY), 0, nRangeSize);
                if(checkAngle(nRangeSize, 180))
                    isFront = true;
                group.push_back(nRangeSize);
                if(checkAngle(nRangeSize, 180))
                    display.at<Vec3b>(y, x) = Vec3b(255, 255, 0);
                else
                    display.at<Vec3b>(y, x) = Vec3b(80, 80, 80);   
            } else {
                nRangeSize++;
                break;
            }
        } else {
            nRangeSize++;
            break;
        }
        
        prevX = nowX;
        prevY = nowY;
    }

    for(int i=0; i<nRangeSize; i++) {
        double nowX = laserScanXY[i][0];
        double nowY = laserScanXY[i][1];
        int x = imageHalfSize[0] + cvRound((nowX/dMaxDist)*imageHalfSize[0]);
        int y = imageHalfSize[1] + cvRound((nowY/dMaxDist)*imageHalfSize[1]);
        if(x >= 0 && x < display.cols && y >= 0 && y < display.rows) {
            if(i == 0 || isSame >= sqrt((prevX-nowX)*(prevX-nowX)+(prevY-nowY)*(prevY-nowY))) {
                if(checkAngle(i, 180))
                    isFront = true;
                if(checkAngle(i, 60) && (get<0>(nearest) < 0 || get<0>(nearest) > sqrt(nowX*nowX+nowY*nowY)))
                    nearest = make_tuple(sqrt(nowX*nowX+nowY*nowY), cnt, i);
                group.push_back(i);
            } else {
                if(isFront) {
                    obstacles.push_back(group);
                    cnt++;
                }
                if(checkAngle(i, 180))
                    isFront = true;
                else
                    isFront = false;
                group.clear();
                group.push_back(i);
                if(checkAngle(i, 60) && (get<0>(nearest) < 0 || get<0>(nearest) > sqrt(nowX*nowX+nowY*nowY)))
                    nearest = make_tuple(sqrt(nowX*nowX+nowY*nowY), cnt, i);
            }
            if(checkAngle(i, 180)) {
                switch(cnt % 3) {
                case 0:
                    display.at<Vec3b>(y, x) = Vec3b(255, 255, 0);
                    break;
                case 1:
                    display.at<Vec3b>(y, x) = Vec3b(255, 0, 255);
                    break;
                default:
                    display.at<Vec3b>(y, x) = Vec3b(0, 255, 255);
                    break;
                }
            } else
                display.at<Vec3b>(y, x) = Vec3b(80, 80, 80);
        }
        prevX = nowX;
        prevY = nowY;
    }

    if(group.size())
        obstacles.push_back(group);

    bool isDangerous = false;
    if(nRangeSize > 0 && nearestMin >= get<0>(nearest)){
        int nearestObstacle = get<1>(nearest);
        for(auto it = obstacles[nearestObstacle].begin(); it != obstacles[nearestObstacle].end(); ++it) {
            if(checkAngle(*it, 60)) {
                double nowX = laserScanXY[*it][0];
                double nowY = laserScanXY[*it][1];
                if(nearestMin >= sqrt(nowX*nowX+nowY*nowY)){
                    int x = imageHalfSize[0] + cvRound((nowX/dMaxDist)*imageHalfSize[0]);
                    int y = imageHalfSize[1] + cvRound((nowY/dMaxDist)*imageHalfSize[1]);
                    display.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
                    isDangerous = true;
                }
            }
        }
    }

    return make_pair(obstacles.size(), (isDangerous ? get<0>(nearest) : -1.));
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
void
printOdometryInfo(nav_msgs::Odometry &odom)
{
    // Display /odom part!
    const ros::Time timestamp = odom.header.stamp;
    const string frame_id = odom.header.frame_id;
    const string child_frame_id = odom.child_frame_id;
    const geometry_msgs::Point translation = odom.pose.pose.position;
    const geometry_msgs::Quaternion rotation = odom.pose.pose.orientation;

    printf("frame_id = %s, child_frame_id = %s\n", frame_id.c_str(), child_frame_id.c_str());
    printf("secs: %d / nsecs: %d\n", timestamp.sec, timestamp.nsec);
    printf("translation = %lf %lf %lf\n", translation.x, translation.y, translation.z);
    printf("rotation = %lf %lf %lf %lf\n\n\n", rotation.x, rotation.y, rotation.z, rotation.w);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
int main(int argc, char **argv)
{
    bool first = true;
    // Initialize the ROS system
    ros::init(argc, argv, "turtle_position_lrf_view");
    ros::NodeHandle nh;

    // Create subscriber objects
    ros::Subscriber subOdom = nh.subscribe("/odom", 100, &odomMsgCallback);
    ros::Subscriber subScan = nh.subscribe("/scan", 10, &scanMsgCallback);
    ros::Subscriber subResult = nh.subscribe("/result", 5, &resultMsgCallback);

    // Display buffer
    Mat display;
    initGrid(display, 801);

    // Odometry buffer
    nav_msgs::Odometry odom;

    // Scan buffer
    sensor_msgs::LaserScan scan;

    // 이동 및 회전 정보
    Vec3d xyz, rpy;

    // 이동궤적

    // LRF scan 정보
    vector<Vec3d> laserScanXY;

    // Mat distance for grid
    const double dGridMaxDist = 4.0;

    // main loop
    while(ros::ok()) {
        // callback 함수을 call!
        ros::spinOnce();

        // receive the global '/odom' message with the mutex
        mutex[0].lock(); {
           odom = g_odom;
        } mutex[0].unlock();

        // odom으로부터 이동 및 회전정보 획득
        convertOdom2XYZRPY(odom, xyz, rpy);

        // 현재의 위치를 저장
        saveCurrentPosition(xyz, trajectory, 0.02);

        // receive the global '/scan' message with the mutex
        mutex[1].lock(); {
           scan = g_scan;
        } mutex[1].unlock();

        // scan으로부터 Cartesian X-Y scan 획득
        convertScan2XYZs(scan, laserScanXY);

        // trajectory를 상대좌표계로 변환
        auto nowTra = transform(trajectory, xyz[0], xyz[1], 2 * M_PI - rpy[2]);

        // 현재 상황을 draw할 display 이미지를 생성
        initGrid(display, 801);
        drawTrajectory(display, nowTra, dGridMaxDist);
        drawCurrentPositionWithRotation(display, xyz, rpy, dGridMaxDist);
        auto ret = drawLRFScan(display, laserScanXY, dGridMaxDist);
        int obstacleSize = ret.first;
        double nearest = ret.second;

        // 2D 영상좌표계에서 top-view 방식의 3차원 월드좌표계로 변환
        transpose(display, display);  // X-Y축 교환
        flip(display, display, 0);  // 수평방향 반전
        flip(display, display, 1);  // 수직방향 반전

        string msg;
        msg = "Number of obstacles: " + to_string(obstacleSize);
        putText(display, msg, Point(10, 30), 0, 0.7, CV_RGB(255, 255, 255));
        if(nearest > 0){
            msg = "The shortest distance to obstacle: " + to_string(nearest);
            putText(display, msg, Point(10, 55), 0, 0.7, CV_RGB(255, 255, 255));
        }

        // 영상 출력!
        imshow("KNU ROS Lecture >> turtle_position_lrf_view", display);
        printOdometryInfo(odom);

        // 사용자의 키보드 입력을 받음!
        int nKey = waitKey(30) % 255;

        if(nKey == 27) {
            // 종료
            break;
        }

        if(nKey == 'c' || nKey == 'C') {
            trajectory.clear();
        }
    }

    return 0;
}

