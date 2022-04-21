#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <string>
#include <fstream>
#include <math.h>


using namespace std;
using namespace cv;
using namespace dnn;

string DIR_PATH = "/Users/markusjonek/Documents/golf-analys";

struct Box_values {
    int x, y;
};

vector<string> get_class_objects() {
    vector<string> class_objects;
    ifstream ifs(string(DIR_PATH + "/data/object_detection_classes_coco.txt").c_str());
    string line;
    while (getline(ifs, line)){
        class_objects.push_back(line);
    }
    return class_objects;
}

Mat object_detection(const Mat& frame){
    auto model = readNet(DIR_PATH + "/data/frozen_inference_graph.pb", DIR_PATH + "/data/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", "TensorFlow");
    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),true, false);
    model.setInput(blob);
    Mat output = model.forward();
    return output;
}

Box_values get_box(Mat frame) {
    struct Box_values box{};
        box.x = 0;
        box.y = 0;

    Mat model_output = object_detection(frame);
    Mat detectionMat(model_output.size[2], model_output.size[3], CV_32F, model_output.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        int class_id = (int)detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);
        string current_object = get_class_objects()[class_id - 1];

        if (confidence > 0.6 && current_object == "person") {
            box.x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            box.y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
        }
    }
    return box;
}

void video_writer(const string& video_path) {
    Mat frame;
    VideoCapture video(video_path);
    video >> frame;

    Box_values box = get_box(frame);

    double fps;
    if (video.get(CAP_PROP_FPS) > 200) {
        fps = 240.0;
    }
    else {
        fps = 15.0;
    }

    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');

    string analyzed_video_path = DIR_PATH + "/analyzed_swing.mp4";

    VideoWriter writer;
    writer.open(analyzed_video_path, codec, fps, frame.size(), true);

    int line_thickness = int(sqrt(frame.cols * frame.rows) / 192);

    while (video.read(frame)) {
        line(frame, Point(0, box.y), Point(frame.cols, box.y), Scalar(0, 0, 255), line_thickness);
        line(frame, Point(box.x, 0), Point(box.x, frame.rows), Scalar(0, 0, 255), line_thickness);
        writer.write(frame);
    }
}

int main(int argc, char *argv[]){
    if (argv[1][0] == '/') {
        video_writer(argv[1]);
    } 
    else {
        string vid = argv[1];
        video_writer(DIR_PATH + "/golf_videos/" + vid);
    }
    system(("open " + DIR_PATH + "/analyzed_swing.mp4").c_str());
    return 0;
}




