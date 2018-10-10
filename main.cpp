#define _DEBUG

#include <cstdlib>
#include <iostream>
#include <vector>
#include <windows.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "GPIOlib.h"

#define PI 3.14159265
#define E 0.1

using namespace cv;
using namespace std;
using namespace GPIO;

//const string CAM_PATH = "/dev/video0";
const string CAM_PATH = "0";
const string MAIN_WINDOW_NAME = "View";
const string CANNY_WINDOW_NAME = "Canny";

const int CANNY_LOWER_BOUND = 50;
const int CANNY_UPPER_BOUND = 250;
const int HOUGH_THRESHOLD = 150;

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
#define SET_SPEED 100
#define HAVE_NEW_VELOCITY 0x01

static PID lPID, rPID;
static PID *lptr = &lPID;
static PID *rptr = &rPID;
int leftSpeed;
int rightSpeed;
int flag;

void readSpeed() {
    resetCounter();
    delay(1000);
    getCounter(&leftSpeed, &rightSpeed);
    flag |= HAVE_NEW_VELOCITY;
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

int main() {
    //init
    init();
    incPIDInit(lptr);
    incPIDInit(rptr);
    leftSpeed = 0;
    rightSpeed = 0;
    flag = 0;

    VideoCapture capture(CAM_PATH);
    //If this fails, try to open as a video camera, through the use of an integer param
    if (!capture.isOpened()) {
        capture.open(atoi(CAM_PATH.c_str()));
    }
    //The time for camera's initialization
    Sleep(1000);

    Mat image;
    while (true) {

        readSpeed();
        if (flag & HAVE_NEW_VELOCITY) {
            controlLeft(FORWARD, incPIDCalc(lptr, leftSpeed));
            controlRight(FORWARD, incPIDCalc(rptr, rightSpeed));
            flag &= ~HAVE_NEW_VELOCITY;
        }

        capture >> image;
        if (image.empty())
            break;
        //Set the ROI(Region Of Interest) for the image
        Rect roi(0, image.rows / 3, image.cols, image.rows / 3);
        Mat imgROI = image(roi);

        //Canny algorithm
        Mat contours;
        Canny(imgROI, contours, CANNY_LOWER_BOUND, CANNY_UPPER_BOUND);
#ifdef _DEBUG
        imshow(CANNY_WINDOW_NAME, contours);
#endif

        vector<Vec2f> lines;
        HoughLines(contours, lines, 1, PI / 180, HOUGH_THRESHOLD);
        Mat result(imgROI.size(), CV_8U, Scalar(255));
        imgROI.copyTo(result);

        //find out the left and right boundaries
        Vec2f leftBound, rightBound;
        bool hasLeft = false, hasRight = false;
        double minLeft = result.cols / 2.0;
        double minRight = result.cols / 2.0;
        for (vector<Vec2f>::const_iterator it = lines.begin(); it != lines.end(); ++it) {
            float rho = (*it)[0];            //First element is distance rho
            float theta = (*it)[1];        //Second element is angle theta

            //Filter to find out lines left and right, and atan(0.09) equals about 5 degrees
            if (theta > 0.09 && theta < 1.48 && rho < minRight) {
                hasRight = true;
                minRight = rho;
                rightBound = *it;
            } else if (theta > 1.62 && theta < 3.05 && rho < minLeft) {
                hasLeft = true;
                minLeft = rho;
                leftBound = *it;
            }
        }

#ifdef _DEBUG
        if (hasLeft) {
            //point of intersection of the line with first row
            Point2f pt1(leftBound[0] / cos(leftBound[1]), 0);
            //point of intersection of the line with last row
            Point2f pt2((leftBound[0] - result.rows * sin(leftBound[1])) / cos(leftBound[1]), result.rows);
            //Draw a line
            line(result, pt1, pt2, Scalar(0, 255, 255), 3, CV_AA);
        }
        if (hasRight) {
            //point of intersection of the line with first row
            Point2f pt1(rightBound[0] / cos(rightBound[1]), 0);
            //point of intersection of the line with last row
            Point2f pt2((rightBound[0] - result.rows * sin(rightBound[1])) / cos(rightBound[1]), result.rows);
            //Draw a line
            line(result, pt1, pt2, Scalar(0, 255, 255), 3, CV_AA);
        }
        imshow(MAIN_WINDOW_NAME, result);
#endif

        //decide directions
        if (hasLeft && hasRight) {
            Point2f left1(leftBound[0] / cos(leftBound[1]), 0);
            Point2f right1(rightBound[0] / cos(rightBound[1]), 0);
            Point2f left2((leftBound[0] - result.rows * sin(leftBound[1])) / cos(leftBound[1]), result.rows);
            Point2f right2((rightBound[0] - result.rows * sin(rightBound[1])) / cos(rightBound[1]), result.rows);

            if (fabs(left1.x - left2.x) < E && fabs(right1.x - right2.x) < E) { //两条垂直线
                turnTo(0);
            } else if (fabs(left1.x - left2.x) < E) { //左边界为垂直线
                double angel = atan(fabs((right1.x - right2.x) / (right1.y - right2.y)));
                angel = 0 - (angel > 45 ? 45 : angel);
                turnTo((int) angel);
            } else if (fabs(right1.x - right2.x) < E) { //右边界为垂直线
                double angel = atan(fabs((right1.x - right2.x) / (right1.y - right2.y)));
                angel = angel > 45 ? 45 : angel;
                turnTo((int) angel);
            } else {
                double k1 = (left1.y - left2.y) / (left1.x - left2.x);
                double k2 = (right1.y - right2.y) / (right1.x - right2.x);
                if (fabs(k1 - k2) < E) {
                    //两条平行线，由于上方已过滤，不可能存在
                } else {
                    //求交点
                    double x = ((right1.y - left1.y) - (k2 * right1.x - k1 * left1.x) / (k1 - k2));
                    double y = k1 * (x - left1.x) + left1.y;
                    //求角平分线
                    double left_len = sqrt(pow(x - left2.x, 2) + pow(y - left2.y, 2));
                    double right_len = sqrt(pow(x - right2.x, 2) + pow(y - right2.y, 2));
                    double prop = left_len / right_len;
                    double x2 = (right2.x - x) * prop + x;
                    double y2 = (right2.y - y) * prop + y;
                    double angel = atan(fabs((x - x2) / (y - y2)));
                    angel = angel > 45 ? 45 : angel;
                    if (x < x2)
                        angel = 0 - angel;
                    turnTo((int) angel);
                }
            }
        } else if (hasLeft) {
            turnTo(45);
        } else if (hasRight) {
            turnTo(-45);
        } else {
            break;
        }

        lines.clear();
        waitKey(1);
    }
    stopLeft();
    stopRight();
    return 0;
}