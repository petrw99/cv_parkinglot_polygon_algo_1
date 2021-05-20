#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <plib/showimages2/showimages2.hpp>

using namespace cv::ximgproc;

using namespace std;

static void draw_delaunay( cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color ) {
    vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<cv::Point> pt(3);
    cv::Size size = img.size();
    cv::Rect rect(0,0, size.width, size.height);
    for( size_t i = 0; i < triangleList.size(); i++ ) {
        cv::Vec6f t = triangleList[i];
        pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5])); // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
            const int len_th = 16;
            double l0 = std::sqrt((pt[1].x-pt[0].x)*(pt[1].x-pt[0].x)+(pt[1].y-pt[0].y)*(pt[1].y-pt[0].y));
            double l1 = std::sqrt((pt[2].x-pt[1].x)*(pt[2].x-pt[1].x)+(pt[2].y-pt[1].y)*(pt[2].y-pt[1].y));
            double l2 = std::sqrt((pt[0].x-pt[2].x)*(pt[0].x-pt[2].x)+(pt[0].y-pt[2].y)*(pt[0].y-pt[2].y));
            if(l0 < len_th) line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            if(l1 < len_th) line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            if(l2 < len_th) line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}


int main()
{
    cv::Mat img = cv::imread("/home/peter/tmp/parkinglot_dots_6_open.png", 0);
//    cv::Mat img = cv::imread("/home/peter/Downloads/OpenCV/opencv-3.4.7/samples/data/building.jpg", 0);

    cv::Mat img_c3;
    cv::cvtColor(img, img_c3, CV_GRAY2BGR);

    const string WinName = "Main Window";
    cv::namedWindow(WinName, CV_WINDOW_NORMAL);

    cv::imshow(WinName, img);
    cv::waitKey(0);

    // Get parking points
    std::vector<cv::Point2f> points;
//    img.forEach<uchar>([&points](uchar &pxl, const int* pos){
//        if(pxl > 0){
//            points.emplace_back(pos[1], pos[0]);
//        }
//    });
    const int len = img.cols*img.rows;
    for(int i=0; i<len; i++){
        int x = i%img.cols;
        int y = i/img.cols;
        if(img.at<uchar>(y,x) > 0) points.emplace_back(x,y);
    }


    cv::Mat canvas;
    cv::cvtColor(img, canvas, CV_GRAY2BGR);
//    cv::Mat canvas = cv::Mat::zeros(img.size(), CV_8UC3);
    for(cv::Point2f it:points){
        cv::circle(canvas, it, 1, cv::Scalar(50,200,100), -1);
    }
    cv::imshow(WinName, canvas);
    cv::waitKey(0);


    cv::Mat mask_8b = img.clone();

//    vector<vector<cv::Point>> cntrs;
//    cv::findContours(mask_8b, cntrs, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
//    cv::RNG rng(12345);
//    for(int i=0; i<cntrs.size(); i++){
//        cv::Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
//        cv::drawContours(canvas, cntrs, i, color);
//    }

//
// Delaunay Triangulation
//
    cv::Size size = mask_8b.size();
    cv::Rect rect = cv::boundingRect(points);;
    cv::Subdiv2D subdiv(rect);
    for( std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++) {
        subdiv.insert(*it);
    }
//    cv::rectangle(canvas, rect, cv::Scalar(50,200,100));
    cv::Scalar delaunay_color(255,255,255);
    draw_delaunay( canvas, subdiv, delaunay_color );

// Skeleton
//    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(9,9));
//    cv::morphologyEx(mask_8b, canvas, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1));
//    cv::imshow(WinName, canvas);
//    cv::waitKey(0);

//    bool useRefine = true;  // LineSegmentDetector
////    cv::Ptr<cv::ximgproc::LineSegmentDetector > ls = useRefine ? createLineSegmentDetector(cv::LSD_REFINE_STD) : createLineSegmentDetector(cv::LSD_REFINE_NONE);
//    cv::Ptr<cv::ximgproc::FastLineDetector> ls = cv::ximgproc::createFastLineDetector();
//    vector<cv::Vec4f> lines_std;
//    ls->detect(mask_8b, lines_std);
//    ls->drawSegments(canvas, lines_std);


    // Standard Hough Line Transform
//    vector<cv::Vec2f> lines; // will hold the results of the detection
//    HoughLines(mask_8b, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
//    // Draw the lines
//    for( size_t i = 0; i < lines.size(); i++ )
//    {
//        float rho = lines[i][0], theta = lines[i][1];
//        cv::Point pt1, pt2;
//        double a = cos(theta), b = sin(theta);
//        double x0 = a*rho, y0 = b*rho;
//        pt1.x = cvRound(x0 + 1000*(-b));
//        pt1.y = cvRound(y0 + 1000*(a));
//        pt2.x = cvRound(x0 - 1000*(-b));
//        pt2.y = cvRound(y0 - 1000*(a));
//        line( canvas, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
//    }


    // Probabilistic Line Transform
//    vector<cv::Vec4i> linesP; // will hold the results of the detection
//    HoughLinesP(mask_8b, linesP, 1, CV_PI/180, 3, 3, 20 ); // runs the actual detection
//    // Draw the lines
//    for( size_t i = 0; i < linesP.size(); i++ )
//    {
//        cv::Vec4i l = linesP[i];
//        line( canvas, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,255,255), 1, cv::LINE_AA);
//    }

//    mask_8b = canvas.clone();
//    cv::cvtColor(mask_8b, mask_8b, cv::COLOR_BGR2GRAY);
//    HoughLinesP(mask_8b, linesP, 1, CV_PI/180, 2, 2, 15 ); // runs the actual detection
//    // Draw the lines
//    for( size_t i = 0; i < linesP.size(); i++ )
//    {
//        cv::Vec4i l = linesP[i];
//        line( canvas, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,0,255), 1, cv::LINE_AA);
//    }


//    std::vector<cv::Point2f> points2;
//    for(int i=0; i<points.size(); i++){

//    }

    cv::Mat mask2;
    cv::cvtColor(canvas, mask2, CV_BGR2GRAY);

    cv::Mat canvas2 = cv::Mat::zeros(canvas.size(), CV_8UC3);

    vector<vector<cv::Point>> cntrs;
    cv::findContours(mask2, cntrs, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    cv::RNG rng(12345);
//    for(int i=0; i<cntrs.size(); i++){
//        if(cntrs.at(i).size() < 30) continue;
//        cv::Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
//        cv::drawContours(canvas2, cntrs, i, color);
//    }

    vector<vector<cv::Point>> approxRect(cntrs.size());

    cv::approxPolyDP(points, approxRect[0], 180, true);
    cv::drawContours(canvas, cntrs, 0, cv::Scalar(50,200,100));

//    cv::approxPolyDP(cntrs, approxRect, 30, true);
    for(int i=1; i<approxRect.size(); i++){
//        if(approxRect.at(i).size() < 30) continue;
        cv::approxPolyDP(cntrs[i], approxRect[i], 12, true);
        cv::Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
        cv::drawContours(canvas2, approxRect, i, color);
    }

    cv::imshow(WinName, canvas2);
    cv::waitKey(0);

    showImages_Vec3b("", 3,3,1, 1, 0, CV_WINDOW_NORMAL, &img_c3, &canvas, &canvas2);

    return 0;
}
