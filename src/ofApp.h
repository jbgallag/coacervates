#pragma once

#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxPanel.h"
#include "ofxGuiGroup.h"
#include "ofxFlocking.h"
//#include "ofxARTTECH3039.h"

class ofApp : public ofBaseApp{

public:
    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);

    void setupGUI();
    void drawGUI();
    //for loading tensorflow model
    void load_model(string model_dir);
    void load_model_index(int index);
    void drawReSampledPolylines(ofPolyline &resampledPoly, int tx, int ty);
    //tensorflow constants
    const int input_shape[2] = {256, 256}; // dimensions {height, width} for input image
    const int output_shape[2] = {256, 256}; // dimensions {height, width} for output image
    const ofVec2f input_range = {-1, 1}; // range of values {min, max} that model expects for input
    const ofVec2f output_range = {-1, 1}; // range of values {min, max} that model outputs
    const string input_op_name = "generator/generator_inputs"; // name of op to feed input to
    const string output_op_name = "generator/generator_outputs"; // name of op to fetch output from

    msa::tf::SimpleModel model;
    ofVideoGrabber videoFeed;
    std::vector<ofImage> captures;
    std::vector<ofImage> prevCaptures;
    std::vector<ofImage> grayScales;
    std::vector<ofFloatImage> coaIn;
    std::vector<ofFloatImage> coaOut;

    std::vector<ofPolyline> polyLines;
    std::vector<ofPolyline> rpolyLines;


    ofFloatImage inImage;
    ofFloatImage outImage;
    ofImage dispImage;
    ofImage drawImage;
    size_t segSizeX,segSizeY,crpX,crpY;
    int vWidth,vHeight;

    //for tensorflow model loading
    ofDirectory models_dir;
    int cur_model_index = 0;

    ofFbo combFBO;
    ofFbo drawFBO;

    bool closePline = false;
    bool rsampPline = false;
    bool tfRdy = false;

    //variables for strange attractor
    float x,y,a,b,c,d;
    int maxIterations;
    int attrIterCount;

    int dispXOff,dispYOff;

    ofParameter<int>scaleRange;
    ofParameter<int>strideRange;
    ofParameter<int>radiusRange;

    ofxPanel rangeGui;

    int startCount = 6;
    ofxFlocking flock;

    ofParameter<float>cohesionDistance;
    ofParameter<float>alignDistance;
    ofParameter<float>separateDistance;



};
