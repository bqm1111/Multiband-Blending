#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;

void createLaplacePyr(cv::Mat img, int num_levels, std::vector<cv::Mat> &pyr)
{
    pyr.resize(num_levels + 1);
    pyr[0] = img.clone();
    for(int i = 0; i < num_levels; i++)
    {
        cv::pyrDown(pyr[i], pyr[i + 1]);
    }
    cv::Mat temp;

    for(int i = 0; i < num_levels; i++)
    {
        cv::pyrUp(pyr[i + 1], temp, pyr[i].size());

        assert(pyr[i].size()==temp.size());
        cv::subtract(pyr[i], temp, pyr[i]);
        cv::Mat show;
        pyr[i].convertTo(show, CV_8U);
    }
}

void restoreImagefromLaplacePyr(std::vector<cv::Mat> &pyr)
{
    if(pyr.empty())
        return;
    cv::Mat temp;
    for(int i = pyr.size() - 1; i > 0; i--)
    {
        printf("Iteration %d\n", i);
        cv::pyrUp(pyr[i], temp, pyr[i - 1].size());
        cv::add(pyr[i - 1], temp, pyr[i - 1]);

    }
}

void fusion(std::vector<cv::Mat>&pyr1, std::vector<cv::Mat>&pyr2, cv::Mat mask1, cv::Mat mask2, std::vector<cv::Mat>&result)
{
    assert(pyr1.size()==pyr2.size());
    int num_level = pyr1.size() - 1;
    std::vector<cv::Mat> mask_pyr1(pyr1.size());
    std::vector<cv::Mat> mask_pyr2(pyr2.size());
    result.resize(num_level +1);
    mask_pyr1[0] = mask1.clone();
    mask_pyr2[0] = mask2.clone();
    cv::Mat m1, m2;

    for(size_t i = 0; i < pyr1.size() - 1; i++)
    {
        cv::pyrDown(mask_pyr1[i], mask_pyr1[i + 1]);
        cv::pyrDown(mask_pyr2[i], mask_pyr2[i + 1]);
        cv::add(pyr1[i]&mask_pyr1[i], pyr2[i] &mask_pyr2[i], result[i]);
    }

    cv::add(pyr1[num_level]&mask_pyr1[num_level], pyr2[num_level]&mask_pyr2[num_level], result[num_level]);
}
int main()
{
    cv::Mat orange, apple;
    orange = cv::imread("../orange.jpg", 0);
    apple = cv::imread("../apple.jpg", 0);
    cv::resize (orange, orange, cv::Size(600, 600));
    cv::resize (apple, apple, cv::Size(600, 600));
    cv::imshow("orange", orange);
//    cv::waitKey(0);
    int num_level = 7;
    int shift =20;
    std::vector<cv::Mat> orangePyr;
    std::vector<cv::Mat> applePyr;
    std::vector<cv::Mat> ret;
    
    cv::Mat mask1(orange.size(), orange.type());
    mask1(cv::Rect(0,0, (mask1.cols + shift)/2, mask1.rows)).setTo (255);
    mask1(cv::Rect((mask1.cols + shift)/2,0,(mask1.cols - shift)/2, mask1.rows)).setTo (0);

    cv::Mat mask2(apple.size(), apple.type());
    mask2(cv::Rect(0,0,(mask2.cols - shift)/2, mask2.rows)).setTo (0);
    mask2(cv::Rect((mask2.cols - shift)/2, 0, (mask2.cols + shift)/2, mask2.rows)).setTo (255);
    createLaplacePyr(orange, num_level, orangePyr);
    createLaplacePyr(apple, num_level, applePyr);
    fusion(orangePyr,applePyr, mask1, mask2, ret);
    restoreImagefromLaplacePyr(ret);
    cv::imshow("ret", ret[0]);
    cv::waitKey(0);

    return 0;
}
