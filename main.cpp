#define _DEBUG
#define _VIDEO

#include <cstdlib>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "GPIOlib.h"

#define E 0.1

using namespace cv;
using namespace std;
using namespace GPIO;

const string CAM_PATH = "/dev/video0";

const int CANNY_LOWER_BOUND = 50;
const int CANNY_UPPER_BOUND = 250;
const int HOUGH_THRESHOLD = 90;

typedef struct PID {
    int set_speed;
    int error;
    int error_next;
    int error_last;
    double A, B, C;
} PID;

#define T 0.05 //采样周期
#define Kp 0.85
#define Ti 0.45 //积分时间
#define Td 0 //微分时间
#define SET_SPEED 5
#define HAVE_NEW_VELOCITY 0x01

static PID lPID, rPID;
static PID *lpid = &lPID;
static PID *rpid = &rPID;
int leftSpeed;
int rightSpeed;
int noLinesCount;
double lastAngle;
int flag;

double distance(Point2f point, double k, double b);

void initialize();

void incPIDInit(PID *sptr);

int incPIDCalc(PID *sptr, int actual_speed);

void forward();

void turn(double angle);

void stop();

int main() {
    initialize();

#ifdef _VIDEO
    VideoCapture capture(CAM_PATH);
    if (!capture.isOpened()) {
        capture.open(atoi(CAM_PATH.c_str()));
    }
#endif

    Mat image;
    while (true) {

#ifdef _VIDEO
        capture >> image;
#endif
        if (image.empty()) {
            break;
        }

        Rect roi(0, image.rows / 2, 4 * image.cols / 5, 2 * image.rows / 4);
        Mat imgROI = image(roi);
        //resize(imgROI, imgROI, Size(0, 0), 0.8, 0.6);

        Mat imgGray;
        cvtColor(imgROI, imgGray, CV_BGR2GRAY);
        Mat imgDilate, dilateElement;
        dilateElement = getStructuringElement(MORPH_RECT, Size(1, 1));
        dilate(imgGray, imgDilate, dilateElement);
        Mat imgErode, erodeElement;
        erodeElement = getStructuringElement(MORPH_RECT, Size(1, 1));
        dilate(imgDilate, imgErode, erodeElement);
        Mat contours;
        Canny(imgErode, contours, CANNY_LOWER_BOUND, CANNY_UPPER_BOUND);
#ifdef _DEBUG
        imshow("Canny", contours);
#endif

        vector<Vec2f> lines;
        HoughLines(contours, lines, 1, CV_PI / 180, HOUGH_THRESHOLD);
        Mat result(imgROI.size(), CV_8U, Scalar(255));
        imgROI.copyTo(result);

        //find out the left and right boundaries
        Vec2f leftBound, rightBound;
        bool hasLeft = false, hasRight = false;
        Point2f center(result.cols / 2, result.rows / 2);
        double minLeft = result.cols / 2.0;
        double minRight = result.cols / 2.0;
        for (vector<Vec2f>::const_iterator it = lines.begin(); it != lines.end(); ++it) {
            float rho = (*it)[0];            //First element is distance rho
            float theta = (*it)[1];        //Second element is angle theta

            Point2f pt1(rho / cos(theta), 0);
            Point2f pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);

            //filter out the vertical or horizontal lines
            if (!(theta > 0.09 && theta < 1.48) && !(theta > 1.62 && theta < 3.05))
                continue;

            double k = (pt2.y - pt1.y) / (pt2.x - pt1.x);
            double b = pt2.y - k * pt2.x;

            double dist = distance(center, k, b);
            double limitK = 0.2;

            if (k < -limitK && dist < minLeft) {
                hasLeft = true;
                minLeft = dist;
                leftBound = *it;
            } else if (k > limitK && dist < minRight) {
                hasRight = true;
                minRight = dist;
                rightBound = *it;
            }

#ifdef _DEBUG
            line(result, pt1, pt2, Scalar(255, 0, 0), 1, CV_AA);
#endif
        }

#ifdef _DEBUG
        if (hasLeft) {
            //point of intersection of the line with first row
            Point2f pt1(leftBound[0] / cos(leftBound[1]), 0);
            //point of intersection of the line with last row
            Point2f pt2((leftBound[0] - result.rows * sin(leftBound[1])) / cos(leftBound[1]), result.rows);
            //Draw a line
            line(result, pt1, pt2, Scalar(0, 255, 255), 2, CV_AA);
        }
        if (hasRight) {
            //point of intersection of the line with first row
            Point2f pt1(rightBound[0] / cos(rightBound[1]), 0);
            //point of intersection of the line with last row
            Point2f pt2((rightBound[0] - result.rows * sin(rightBound[1])) / cos(rightBound[1]), result.rows);
            //Draw a line
            line(result, pt1, pt2, Scalar(0, 255, 255), 2, CV_AA);
        }
#endif

        //decide directions
        double angle = 0;
        if (hasLeft && hasRight) {
            Point2f left1(leftBound[0] / cos(leftBound[1]), 0);
            Point2f right1(rightBound[0] / cos(rightBound[1]), 0);
            Point2f left2((leftBound[0] - result.rows * sin(leftBound[1])) / cos(leftBound[1]), result.rows);
            Point2f right2((rightBound[0] - result.rows * sin(rightBound[1])) / cos(rightBound[1]), result.rows);

            //intersection
            double x = 0, y = 0;
            bool hasIntersection = true;
            if (fabs(left1.x - left2.x) < E && fabs(right1.x - right2.x) < E) { //两条垂直线
                hasIntersection = false;
            } else if (fabs(left1.x - left2.x) < E) { //左边界为垂直线
                double k = (right1.y - right2.y) / (right1.x - right2.x);
                x = left1.x;
                y = k * (x - right1.x) + right1.y;
            } else if (fabs(right1.x - right2.x) < E) { //右边界为垂直线
                double k = (left1.y - left2.y) / (left1.x - left2.x);
                x = right1.x;
                y = k * (x - left1.x) + left1.y;
            } else {
                double k1 = (left1.y - left2.y) / (left1.x - left2.x);
                double k2 = (right1.y - right2.y) / (right1.x - right2.x);
                if (fabs(k1 - k2) < E) {
                    //两条平行线，由于上方已过滤，不可能存在
                    hasIntersection = false;
                } else {
                    x = ((right1.y - left1.y) - (k2 * right1.x - k1 * left1.x) / (k1 - k2));
                    y = k1 * (x - left1.x) + left1.y;
                }
            }

            //求角平分线
            if (hasIntersection) {
//                double left_len = sqrt(pow(x - left2.x, 2) + pow(y - left2.y, 2));
//                double right_len = sqrt(pow(x - right2.x, 2) + pow(y - right2.y, 2));
//                double prop = left_len / right_len;
//                double x2 = (((right2.x - x) * prop + x) + left2.x) / 2;
//                double y2 = (((right2.y - y) * prop + y) + left2.y) / 2;
                Point2f bottomMid(result.cols / 2, result.rows);
                angle = atan(fabs((x - bottomMid.x) / (y - bottomMid.y))) * 180 / CV_PI;

                angle = angle > 45 ? 45 : angle;
                if (x < bottomMid.x)
                    angle = 0 - angle;
#ifdef _DEBUG
                line(result, Point2f(x, y), bottomMid, Scalar(0, 255, 0), 2, CV_AA);
#endif
            } else angle = lastAngle;
            noLinesCount = 0;
        } else if (hasLeft) {
            angle = 15;
            noLinesCount = 0;
        } else if (hasRight) {
            angle = -20;
            noLinesCount = 0;
        } else {
            noLinesCount++;
            if(noLinesCount == 50)
                break;
            angle = 0;
        }

#ifdef _DEBUG
        imshow("View", result);
#endif

        turnTo(angle);
        forward();
        lines.clear();
        waitKey(1);
    }

    stop();
    capture.release();
    return 0;
}

double distance(Point2f point, double k, double b) {
    return fabs(k * point.x - point.y + b) / sqrt(k * k + 1);
}

void initialize() {
    GPIO::init();
    incPIDInit(lpid);
    incPIDInit(rpid);
    leftSpeed = 0;
    rightSpeed = 0;
    noLinesCount = 0;
    lastAngle = 0;
    flag = 0;
}

void incPIDInit(PID *sptr) {
    sptr->set_speed = SET_SPEED;
    sptr->error = 0;
    sptr->error_next = 0;
    sptr->error_last = 0;
    sptr->A = Kp * (1 + T / Ti + Td / T);
    sptr->B = Kp * (1 + 2 * Td / T);
    sptr->C = Kp * Td / T;
}

int incPIDCalc(PID *sptr, int actual_speed) {
    sptr->error = sptr->set_speed - actual_speed;
    double inc_speed, res_speed;
    inc_speed = sptr->A * sptr->error -
                sptr->B * sptr->error_next +
                sptr->C * sptr->error_last;
    res_speed = actual_speed + inc_speed;
    sptr->error_last = sptr->error_next;
    sptr->error_next = sptr->error;
    return (int) res_speed;
}

void forward() {
//    resetCounter();
//    getCounter(&leftSpeed, &rightSpeed);
//    flag |= HAVE_NEW_VELOCITY;
//    if (flag & HAVE_NEW_VELOCITY) {
//        controlLeft(FORWARD, incPIDCalc(lpid, leftSpeed));
//        controlRight(FORWARD, incPIDCalc(rpid, rightSpeed));
//        flag &= ~HAVE_NEW_VELOCITY;
//    }
    controlLeft(FORWARD, SET_SPEED);
    controlRight(FORWARD, SET_SPEED);
}

void turn(double angle) {
    if (angle * lastAngle < 0 || fabs(angle - lastAngle) > 3) {
        turnTo((int)angle);
    }
    lastAngle = angle;
}

void stop() {
    stopLeft();
    stopRight();
}
