#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/text.hpp>

using namespace std;
using namespace cv;

void detect_text(string input) {
    Ptr<text::BaseOCR> orc = text::OCRTesseract::create(nullptr, "rus");
    string text;

    Mat original = imread(input);
    imshow("original", original);
    Mat compressed;
    pyrDown(original, compressed);
    Mat small;
    cvtColor(compressed, small, COLOR_BGR2GRAY);

    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);

    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);

    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));

    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);

        drawContours(mask, contours, idx, Scalar(255, 255, 255), FILLED);

        double r = (double) countNonZero(maskROI) / (rect.width * rect.height);

        if (r > 0.45 && (rect.height > 30 && rect.width > 30)) {
            rectangle(compressed, rect, Scalar(255, 0, 0), 2);
            Mat temp = compressed( Range(rect.y - 5, rect.y + rect.height), Range(rect.x - 5, rect.x + rect.width));
            imshow(to_string(idx), temp);
        }
    }

    orc->run(original, text);
    cout << text << endl;
    waitKey(0);
}

int main(int argc, char *argv[]) {
    detect_text(string("../../5_task/data/test_marker.jpg"));
    return 0;
}
