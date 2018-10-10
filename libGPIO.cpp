//
// Created by 王川源 on 2018/10/10.
//
#include "GPIOlib.h"
#include <iostream>

using namespace std;

int GPIO::init(){
    cout << "GPIO init" << endl;
    return 1;
}

int GPIO::controlLeft(int direction,int speed) {
    cout << "left run " << speed << " " << direction << endl;
    return 1;
}

int GPIO::controlRight(int direction,int speed) {
    cout << "right run " << speed << " " << direction << endl;
    return 1;
}

int GPIO::stopLeft() {
    cout << "left stop" << endl;
    return 1;
}

int GPIO::stopRight() {
    cout << "right stop" << endl;
    return 1;
}

int GPIO::resetCounter() {
    return 1;
}

void GPIO::getCounter(int *countLeft,int *countRight) {
    *countLeft = 100;
    *countRight = 100;
}

int GPIO::turnTo(int angle) {
    cout << "turn to " << angle <<endl;
    return 1;
}

void GPIO::delay(int milliseconds) {

}

