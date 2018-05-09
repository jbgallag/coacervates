#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    //set video w/h
    vWidth = 256;
    vHeight = 256;

    ofSetVerticalSync(true);

    //how many cells to break image into in xy
    segSizeX = 1;
    segSizeY = 1;
    //size of each image
    crpX = (size_t)vWidth/segSizeX;
    crpY = (size_t)vHeight/segSizeY;
    //bools
    rsampPline = true;

    drawImage.allocate(vWidth,vHeight,OF_IMAGE_COLOR);
    
    for(size_t y=0; y<segSizeY; y++) {
        for(size_t x=0; x<segSizeX; x++) {
            ofFloatImage inImg,outImg;
            ofImage anImg;
            outImg.allocate(crpX,crpY,OF_IMAGE_COLOR);
            inImg.allocate(crpX,crpY, OF_IMAGE_COLOR);
            anImg.allocate(crpX,crpY, OF_IMAGE_COLOR);
            coaOut.push_back(outImg);
            coaIn.push_back(inImg);
            grayScales.push_back(anImg);
        }
    }

    drawFBO.allocate(vWidth,vHeight, GL_RGB);
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

    ofPushMatrix();
    drawFBO.begin();
    //ofSetLineWidth(20);
    for(auto &pline: polyLines) {
        pline.draw();
    }
    
    
    for(auto &pline: rpolyLines) {
        drawReSampledPolylines(pline);
    }
    
    drawFBO.end();
    drawFBO.draw(0,0);
    ofPopMatrix();

    //read the draw FBO then break up into segSizeX*segSizeY images
    size_t cpIdx = 0;
    for(size_t y=0; y<segSizeY; y++) {
        for(size_t x=0; x<segSizeX; x++) {
            size_t startX = x*crpX;
            size_t startY = y*crpY;
            coaIn[cpIdx].cropFrom(drawImage, startX, startY, crpX, crpY);
            cpIdx++;
        }
    }

    cpIdx = 0;
    for(size_t y = 0; y<segSizeY; y++) {
        for(size_t x = 0; x<segSizeX; x++) {
            float xStart = x*crpX + vWidth;
            float yStart = y*crpY;

            model.run_image_to_image(coaIn[cpIdx],coaOut[cpIdx], input_range, output_range);
            coaOut[cpIdx].draw(xStart,yStart);
            cpIdx++;
        }
    }
}

void ofApp::drawReSampledPolylines(ofPolyline &resampledPoly)
{
    ofPushMatrix();
    //ofTranslate(ofGetWidth() / 2, 0);
    //ofDrawRectangle(0, 0, ofGetWidth() / 2, ofGetHeight());
    
    
    // Draw the resampled polyline in yellow.
    ofSetColor(ofColor::yellow);
    resampledPoly.draw();
    // Draw its vertices.
    ofSetColor(255, 255, 127);
    for (auto vertex: resampledPoly.getVertices())
    {
        ofDrawCircle(vertex, 3);
    }
    
    ofSetColor(255);
    ofDrawBitmapString("Resampled by Spacing", 14, ofGetHeight() - 14);
    
    float time = ofGetElapsedTimef();
    
    for (std::size_t i = 0; i < resampledPoly.size(); i=i+3)
    {
        float phase = ofMap(i, 0, resampledPoly.size(), 0, glm::pi<float>());
        float scaling = 25 * sin(time + phase);
        
        // The normal is a normalized vector representing the "Normal" direction.
        glm::vec3 theNormal = resampledPoly.getNormalAtIndex(i);
        // First we stretch it to the length we want.
        glm::vec3 theScaledNormal = theNormal * scaling;
        glm::vec3 vertex = resampledPoly[i];
        // Then we translate it to get its position relative to the vertex.
        glm::vec3 positiveNormalVertexOffset = vertex + theScaledNormal;
        glm::vec3 negativeNormalVertexOffset = vertex - theScaledNormal;
        
        ofNoFill();
        //ofSetColor(255, 80);
        ofDrawCircle(positiveNormalVertexOffset, scaling / 3);
        ofDrawLine(vertex, positiveNormalVertexOffset);
        
        //ofSetColor(255, 0, 0, 80);
        ofDrawCircle(negativeNormalVertexOffset, scaling / 3);
        ofDrawLine(vertex, negativeNormalVertexOffset);
    }
    ofPopMatrix();
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
    polyLines[polyLines.size()-1].addVertex(x,y);
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

    ofPolyline pLine;
    ofPolyline rpLine;
    polyLines.push_back(pLine);
    rpolyLines.push_back(rpLine);
    polyLines[polyLines.size()-1].addVertex(x,y);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

    polyLines[polyLines.size()-1].addVertex(x,y);
    //if(closePline)
        polyLines[polyLines.size()-1].close();
    if(rsampPline)
        rpolyLines[rpolyLines.size()-1] = polyLines[polyLines.size()-1].getResampledBySpacing(20);
    
    
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
