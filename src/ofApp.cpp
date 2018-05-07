#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    //set video w/h
    vWidth = 512;
    vHeight = 512;
    vector<ofVideoDevice> devices = videoFeed.listDevices();

    for(size_t i = 0; i < devices.size(); i++){
        if(devices[i].bAvailable){
            //log the device
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName;
        }else{
            //log the device and note it as unavailable
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName << " - unavailable ";
        }
    }
    videoFeed.setDeviceID(0);
    videoFeed.setDesiredFrameRate(15);
    videoFeed.initGrabber(vWidth,vHeight);
    ofSetVerticalSync(true);

    //how many cells to break image into in xy
    segSizeX = 2;
    segSizeY = 2;
    //size of each image
    crpX = (size_t)vWidth/segSizeX;
    crpY = (size_t)vHeight/segSizeY;

    for(size_t y=0; y<segSizeY; y++) {
        for(size_t x=0; x<segSizeX; x++) {
            ofImage anImg,gsImg;
            ofFloatImage inImg,outImg;
            anImg.allocate(crpX,crpY,OF_IMAGE_COLOR);
            gsImg.allocate(crpX,crpY, OF_IMAGE_GRAYSCALE);
            outImg.allocate(crpX,crpY,OF_IMAGE_COLOR);
            inImg.allocate(crpX,crpY, OF_IMAGE_COLOR);
            captures.push_back(anImg);
            grayScales.push_back(gsImg);
            coarImages.push_back(outImg);
            coarIn.push_back(inImg);
        }
    }
    combFBO.allocate(vWidth,vHeight, GL_RGB);
    //load tensorflow model
    models_dir.listDir("models");
    if(models_dir.size()==0) {
        ofLogError() << "Couldn't find models folder." << msa::tf::missing_data_error();
        assert(false);
        ofExit(1);
    }
    models_dir.sort();
    load_model_index(0); // load first model


}
void ofApp::load_model_index(int index) {
    cur_model_index = ofClamp(index, 0, models_dir.size()-1);
    load_model(models_dir.getPath(cur_model_index));
}


//--------------------------------------------------------------
void ofApp::update(){

}

void ofApp::load_model(string model_dir)
{
    ofLogVerbose() << "loading model " << model_dir;

    // init the model
    // note that it expects arrays for input op names and output op names, so just use {}
    model.setup(ofFilePath::join(model_dir, "/graph_frz.pb"), {input_op_name}, {output_op_name});
    if(! model.is_loaded()) {
        ofLogError() << "Model init error.";
        ofLogError() << msa::tf::missing_data_error();
        assert(false);
        ofExit(1);
    }

    // init tensor for input. shape should be: {batch size, image height, image width, number of channels}
    // (ideally the SimpleModel graph loader would read this info from the graph_def and call this internally)
    model.init_inputs(tensorflow::DT_FLOAT, {1, input_shape[0], input_shape[1], 3});
    tfRdy = true;
    printf("DONE WITH TENSORFLOW INIT!!! %s %s\n",model_dir.c_str(),ofFilePath::join(model_dir, "/graph_frz.pb").c_str());
}
//--------------------------------------------------------------
void ofApp::draw(){

    if(videoFeed.isInitialized() && tfRdy) {
        videoFeed.update();
        if(videoFeed.isFrameNew()) {
            size_t cpIdx = 0;
            for(size_t y=0; y<segSizeY; y++) {
                for(size_t x=0; x<segSizeX; x++) {
                    size_t startX = x*crpX;
                    size_t startY = y*crpY;
                   // captures[cpIdx].clear();
                    videoFeed.getPixels().cropTo(captures[cpIdx].getPixels(), startX,startY,crpX,crpY);
                    captures[cpIdx].update();
                   // captures[cpIdx].mirror(false,true);

                   // grayScales[cpIdx].clear();
                    ofxCv::convertColor(captures[cpIdx],grayScales[cpIdx],CV_RGB2GRAY);
                    ofxCv::Canny(grayScales[cpIdx],grayScales[cpIdx],20,20,3);
                    grayScales[cpIdx].update();



                    cpIdx++;
                }
            }

            cpIdx=0;
            for(size_t y=0; y<segSizeY; y++) {
                for(size_t x=0; x<segSizeX; x++) {
                    size_t startX = x*crpX;
                    size_t startY = y*crpY;
                    combFBO.begin();
                        ofSetColor(255);
                        grayScales[cpIdx].draw(0, 0, combFBO.getWidth()/segSizeX, combFBO.getHeight()/segSizeY);
                    combFBO.end();
                    combFBO.readToPixels(coarIn[cpIdx].getPixels());
                    //combFBO.clear();
                    cpIdx++;
                }
            }
        }
    }
    size_t cpIdx = 0;
    for(size_t y = 0; y<segSizeY; y++) {
        for(size_t x = 0; x<segSizeX; x++) {
            float xStart = x*crpX;
            float yStart = y*crpY;
            //combFBO.readToPixels(coarIn[cpIdx].getPixels());
            //run the canny filter img through network
            model.run_image_to_image(coarIn[cpIdx],coarImages[cpIdx], input_range, output_range);
          //  ofPushMatrix();
           // drawImage(coarImages[cpIdx], "");
           // ofPopMatrix();
            coarImages[cpIdx].draw(xStart,yStart);
            cpIdx++;
        }
    }
    //combFBO.draw(600,0);

    cpIdx = 0;
    for(size_t y = 0; y<segSizeY; y++) {
        for(size_t x = 0; x<segSizeX; x++) {
            float xStart = x*crpX + 600;
            float yStart = y*crpY;
           // coarImages[cpIdx].update();
            grayScales[cpIdx].draw(xStart,yStart);
            cpIdx++;
        }
    }
    cpIdx = 0;
    for(size_t y = 0; y<segSizeY; y++) {
        for(size_t x = 0; x<segSizeX; x++) {
            float xStart = x*crpX + 300;
            float yStart = y*crpY + 600;
           // coarImages[cpIdx].update();
            //grayScales[cpIdx].draw(xStart,yStart);
            captures[cpIdx].draw(xStart,yStart);
            cpIdx++;
        }
    }
    //combFBO.draw(0,0);
}

template <typename T> bool ofApp::drawImage(const T& img, string label) {
    if(img.isAllocated()) {
        ofSetColor(255);
        ofFill();
        img.draw(0, 0);

        // draw border
     /*   ofNoFill();
        ofSetColor(200);
        ofSetLineWidth(1);
        ofDrawRectangle(0, 0, img.getWidth(), img.getHeight());

        // draw label
        ofDrawBitmapString(label, 10, img.getHeight()+15);

        ofTranslate(img.getWidth(), 0);*/
        return true;
    }

    return false;
}
//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
