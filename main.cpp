#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include "pcl/filters/filter.h"
#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/io/ply_io.h"
#include "pcl/io/obj_io.h"
#include "pcl/common/common.h"
#include "pcl/surface/poisson.h"
#include "pcl/common/transforms.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "StereoEfficientLargeScale.h"
#include "rtabmap/core/util2d.h"

using namespace std;
using namespace pcl;

cv::Point3f projectDisparityTo3D(const cv::Point2f & pt, float disparity)
{
    if(disparity > 0.0f)
    {
        float W = 0.35 / disparity;
        return cv::Point3f((pt.x - 640) * W, (pt.y - 360) * W, 762.72 * W);
    }
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    return cv::Point3f(bad_point, bad_point, bad_point);
}

Eigen::Matrix3d getRFromrpy(const Eigen::Vector3d& rpy)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d ea(rpy(0),rpy(1),rpy(2));
    R = Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitX());
    return R;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFromDisparityRGB(const cv::Mat & imageRgb,
                                                             const cv::Mat & imageDisparity,
                                                             int decimation)
{
    float maxDepth = 20.0;
    float minDepth = 0.0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);


    //cloud.header = cameraInfo.header;
    cloud->height = imageRgb.rows/decimation;
    cloud->width  = imageRgb.cols/decimation;
    cloud->is_dense = false;
    cloud->resize(cloud->height * cloud->width);

    for(int h = 0; h < imageRgb.rows && h/decimation < (int)cloud->height; h+=decimation)
    {
        for(int w = 0; w < imageRgb.cols && w/decimation < (int)cloud->width; w+=decimation)
        {
            float disp = imageDisparity.at<float>(h,w);
            pcl::PointXYZRGB & pt = cloud->at((h/decimation)*cloud->width + (w/decimation));

            pt.b = imageRgb.at<cv::Vec3b>(h,w)[0];
            pt.g = imageRgb.at<cv::Vec3b>(h,w)[1];
            pt.r = imageRgb.at<cv::Vec3b>(h,w)[2];

            cv::Point3f ptXYZ = projectDisparityTo3D(cv::Point2f(w, h), disp);
            if(ptXYZ.z >= minDepth && ptXYZ.z <= maxDepth)
            {
                pt.x = ptXYZ.x;
                pt.y = ptXYZ.y;
                pt.z = ptXYZ.z;
            }
            else
            {
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return cloud;
}

int main()
{
//    cv::FileStorage fs("/home/uisee/Downloads/stereo_20190710_huadong_16.yaml", cv::FileStorage::READ);
//    cv::Mat K_left, K_right, D_left, D_right, R_left, R_right, P_left, P_right;
//    fs["M1"] >> K_left;
//    fs["M2"] >> K_right;
//    fs["D1"] >> D_left;
//    fs["D2"] >> D_right;
//    fs["R1"] >> R_left;
//    fs["R2"] >> R_right;
//    fs["P1"] >> P_left;
//    fs["P2"] >> P_right;
//    cv::Mat map11, map12;
//    cv::Mat map21, map22;
//    cv::Mat left(720, 1280, CV_8UC1);
//    cv::initUndistortRectifyMap(K_left, D_left, R_left, P_left, left.size(), CV_16SC2, map11, map12);
//    cv::initUndistortRectifyMap(K_right, D_right, R_right, P_right, left.size(), CV_16SC2, map21, map22);
//    ifstream in_image("/home/uisee/workspace/file/build/real.txt");
//    ifstream in_pose("/home/uisee/Data/L1_clock_outer_alone_test.txt.pn");

    ifstream in_image("/home/uisee/workspace/EfficientLargeScaleStereo/stereo1.txt");
    ifstream in_pose("/home/uisee/Data/stereo-0/L1_clock_outer_alone_test.txt.pn");
    ifstream in_ref("/home/uisee/Data/stereo-0/ref_pose.txt");

    string left("/home/uisee/Data/stereo-0/left/");
    string right("/home/uisee/Data/stereo-0/right/");
    //string img("/home/uisee/Data/image_capturer_0/");
    string image;
    Eigen::Matrix<double,1,6> pose;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr Clouds(new pcl::PointCloud<pcl::PointXYZRGB>);
    Eigen::Matrix4f T_cam_car;
    T_cam_car << 1, 0, 0, -0.175,
               0, 0, 1, 1.99,
               0, -1, 0, 0,
               0, 0, 0, 1;
    string startx, starty;
    in_pose >> startx >> starty;
    bool flag = true;
    auto start = std::chrono::system_clock::now();
    while(in_image >> image)
    {
        string x, y, z, roll, pitch, yaw, filename;
        if(!(in_pose >> filename))
            break;
        string x_ref, y_ref, z_ref, roll_ref, pitch_ref, yaw_ref;
        in_ref >> x_ref >> y_ref >> z_ref >> roll_ref >> pitch_ref >> yaw_ref;
        in_pose >> pitch >> yaw >> roll >> x >> z >> y;
        flag = !flag;
        //if(flag)
            //continue;
        pose << stod(x_ref), stod(y_ref), stod(z_ref), stod(roll_ref), stod(pitch_ref), stod(yaw_ref);
        //pose << stod(x), stod(y), -stod(z), -stod(roll), stod(pitch), -stod(yaw);//-stod(roll), -stod(pitch), -stod(yaw);
        Eigen::Matrix3d R = getRFromrpy(Eigen::Vector3d(pose(3), pose(4), pose(5)));
        Eigen::Matrix4d T;
        T << R(0, 0), R(0, 1), R(0, 2), pose(0),
             R(1, 0), R(1, 1), R(1, 2), pose(1),
             R(2, 0), R(2, 1), R(2, 2), pose(2),
             0.0, 0.0, 0.0, 1.0;
        Mat leftim = imread(left + image);
        Mat rightim = imread(right + image);
//        Mat im = imread(img + image);
//        Mat rightim = im(cv::Rect(im.cols / 2, 0, im.cols / 2, im.rows));//imread(left + image);
//        Mat leftim = im(cv::Rect(0, 0, im.cols / 2, im.rows));//imread(right + image);
        Mat leftgray, rightgray;
        cv::cvtColor(leftim, leftgray, CV_BGR2GRAY);
        cv::cvtColor(rightim, rightgray, CV_BGR2GRAY);
//        cv::cvtColor(leftim, leftgray, CV_BGR2GRAY);
//        cv::cvtColor(rightim, rightgray, CV_BGR2GRAY);
//        cv::remap(leftgray, leftgray, map11, map12, INTER_LINEAR);
//        cv::remap(rightgray, rightgray, map21, map22, INTER_LINEAR);

        Mat dest;
        StereoEfficientLargeScale elas(0,128);
        elas(leftgray,rightgray,dest,100);

        //dest = rtabmap::util2d::disparityFromStereoImages(leftgray, rightgray);
        dest.convertTo(dest,CV_32FC1,1.0/16);
//        Mat show;
//        dest.convertTo(show,CV_8U,1.0/8);
//        imshow("disp",show);
//        waitKey();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudlocal(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloudlocal = cloudFromDisparityRGB(leftim, dest, 4);
        std::vector<int> index;
        pcl::removeNaNFromPointCloud(*cloudlocal, *cloudlocal, index);
        pcl::transformPointCloud(*cloudlocal, *cloudlocal, T_cam_car);
        pcl::transformPointCloud(*cloudlocal, *cloudlocal, T);

        *Clouds += *cloudlocal;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(Clouds);
    statistical_filter.filter(*tmp);

//    statistical_filter.setInputCloud(tmp);
//    statistical_filter.filter(*Clouds);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The exporting costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;

    std::cout << "Saving cloud_elas_ref_half_SOR.ply... (" << static_cast<int>(tmp->size()) << " points)" << std::endl;
    pcl::PLYWriter writer;
    writer.write("cloud_elas_ref_half_SOR.ply", *tmp);
    std::cout << "Saving cloud_elas_ref_half_SOR.ply... done!" << std::endl;


	
	

//    Mat show;
//    dest.convertTo(show,CV_8U,1.0/8);
//    //dest.convertTo(show,CV_32FC1,1.0/16);
//    std::cout << show.at<float>(520, 36);
//    imshow("disp",show);
//    waitKey();
    return 0;
}
