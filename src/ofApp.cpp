#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    //set video w/h
    ofBackground(0);
    ofEnableAlphaBlending();
    ofEnableBlendMode(OF_BLENDMODE_SCREEN);
    vWidth = 1536;
    vHeight = 768;

    dispXOff = 192;
    dispYOff = 216;

    ofSetVerticalSync(true);

    //how many cells to break image into in xy
    segSizeX = 1;
    segSizeY = 1;
    //size of each image
    crpX = 256;//(size_t)vWidth/segSizeX;
    crpY = 256;//(size_t)vHeight/segSizeY;
    //bools
    rsampPline = true;
    closePline = true;

    dispImage.allocate(vWidth,vHeight,OF_IMAGE_COLOR);
    outImage.allocate(crpX,crpY,OF_IMAGE_COLOR);
    inImage.allocate(crpX,crpY,OF_IMAGE_COLOR);

    dispImageTwo.allocate(crpX,crpY,OF_IMAGE_COLOR);
    dispImageThree.allocate(crpX,crpY,OF_IMAGE_COLOR);
    outImageTwo.allocate(crpX,crpY,OF_IMAGE_COLOR);
    inImageTwo.allocate(crpX,crpY,OF_IMAGE_COLOR);

    testImage.allocate(crpX,crpY,OF_IMAGE_COLOR);
    testImageTwo.allocate(crpX,crpY,OF_IMAGE_COLOR);

   // testImage.load("0618.png");
    drawImage.allocate(vWidth/2,vHeight,OF_IMAGE_COLOR);
    drawImageTwo.allocate(vWidth/2,vHeight,OF_IMAGE_COLOR);
    /*for(size_t y=0; y<segSizeY; y++) {
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
    }*/

    drawFBO.allocate(vWidth*2,vHeight, GL_RGB);
   // drawFBOTwo.allocate(vWidth,vHeight, GL_RGB);
    //load tensorflow model
    models_dir.listDir("models");
    if(models_dir.size()==0) {
        ofLogError() << "Couldn't find models folder." << msa::tf::missing_data_error();
        assert(false);
        ofExit(1);
    }
    models_dir.sort();
    load_model_index(0); // load first model
    load_model_indexTwo(6);

    drawTest = true;
    scaleOsc = 1.0;
    scaleCount = 0;
    scaleCountMax = 2000;

    frameCount = 1;
    frameCountMax = 2312;
    frameCountOffset = ofRandom(100,1500);
    setupGUI();
    //for (int i = startCount; i > 0; i--) {
     //       flock.addBoid();
     //   }


    freeDraw = true;
    flockDraw = false;


}
void ofApp::load_model_index(int index) {
    cur_model_index = ofClamp(index, 0, models_dir.size()-1);
    load_model(models_dir.getPath(cur_model_index));
}

void ofApp::load_model_indexTwo(int index) {
    cur_model_index = ofClamp(index, 0, models_dir.size()-1);
    load_modelTwo(models_dir.getPath(cur_model_index));
}

//--------------------------------------------------------------
void ofApp::update(){
    //flock.update();

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

void ofApp::load_modelTwo(string model_dir)
{
    ofLogVerbose() << "loading model " << model_dir;

    // init the model
    // note that it expects arrays for input op names and output op names, so just use {}
    modelTwo.setup(ofFilePath::join(model_dir, "/graph_frz.pb"), {input_op_name}, {output_op_name});
    if(! modelTwo.is_loaded()) {
        ofLogError() << "Model init error.";
        ofLogError() << msa::tf::missing_data_error();
        assert(false);
        ofExit(1);
    }

    // init tensor for input. shape should be: {batch size, image height, image width, number of channels}
    // (ideally the SimpleModel graph loader would read this info from the graph_def and call this internally)
    modelTwo.init_inputs(tensorflow::DT_FLOAT, {1, input_shape[0], input_shape[1], 3});
    tfRdy = true;
    printf("DONE WITH TENSORFLOW INIT!!! %s %s\n",model_dir.c_str(),ofFilePath::join(model_dir, "/graph_frz.pb").c_str());
}
//--------------------------------------------------------------
void ofApp::draw(){

    drawGUI();

    drawFBO.begin();
    ofClear(0,0,0);
    drawFBO.end();
    drawFBO.begin();
  /*  if(drawTestImage) {
            testImage.clear();
            testImage.load(getImageFileName(frameCount));
            testImage.resize(vWidth,vHeight);
            testImage.draw(0,0);
            testImageTwo.clear();
            testImageTwo.load(getImageFileName(frameCount+frameCountOffset));
            testImageTwo.resize(vWidth,vHeight);
            testImageTwo.draw(0,0);
            frameCount++;
            if((frameCount + frameCountOffset) > frameCountMax) {
                frameCount = ofRandom(100,500);
                frameCountOffset = ofRandom(100,1500);
            }

    }*/

    if(freeDraw) {
        ofPushStyle();
        ofSetColor(175);
        ofSetLineWidth(fdLineWidth);
        for(auto &pline: polyLines) {
            pline.draw();
        }
        ofPopStyle();
    }

    if(flockDraw) {
        //draw pline while available
        for(auto &pline: polyLines) {
            pline.draw();
        }
        if(flock.boids.size() > 0) {
            for(size_t i = 0; i<flock.boids.size(); i++) {
                flock.boids[i].setSeperateDistance(separateDistance);
                flock.boids[i].setCohesionDistance(cohesionDistance);
                flock.boids[i].setAlignDistance(alignDistance);
                flock.boids[i].setMaxSpeed(maxSpeed);
            }
            flock.update();
        }
        flock.draw();
        if(flock.boids.size() > 0)
            drawFlockingPolylines();
    }
   // ofPopMatrix();
   // flock.draw();
   // if(rsampPline) {
    /*for(size_t i = 0; i<flock.boids.size(); i++) {
        ofPushMatrix();
        ofTranslate(flock.boids[i].loc.x,flock.boids[i].loc.y);
        drawReSampledPolylines(rpolyLines[i],0,0);
        ofPopMatrix();
    }*/
    /*if(flock.boids.size() > 0) {
        size_t cnt = 0;
        for(size_t i = 0; i<rpolyLines.size(); i++) {

            //if(flock.boids.size() > 0) {
                ofPushMatrix();
                //ofTranslate(flock.boids[cnt].lastLoc.x-dispXOff,flock.boids[cnt].lastLoc.y-dispYOff);

                drawReSampledPolylines(rpolyLines[i],flock.boids[cnt].lastLoc.x,flock.boids[cnt].lastLoc.y);
               // ofTranslate(-1.0*flock.boids[cnt].lastLoc.x-dispXOff,-1.0*flock.boids[cnt].lastLoc.y-dispYOff);
               // ofTranslate(-dispXOff,-dispYOff);
                ofPopMatrix();
             //   printf("FLOCK_LOX: %d %f %f\n",cnt,flock.boids[cnt].loc.x,flock.boids[cnt].loc.y);
                printf("FLOCK_LOX: %d %f %f\n",cnt,flock.boids[cnt].lastLoc.x,flock.boids[cnt].lastLoc.y);
                cnt++;
            }
           // drawReSampledPolylines(pline);

            //cnt++;
        }
   // }*/

   // for(auto &pline: rpolyLines) {
    //    drawReSampledPolylines(pline,0,0);
  //  }
   // ofPopMatrix();
    drawFBO.end();
   // drawFBO.draw(dispXOff,dispYOff);
   // ofPopMatrix();



    drawFBO.readToPixels(dispImage.getPixels());
    dispImage.update();

    dispImageTwo.cropFrom(dispImage,0,0,vWidth/2,vHeight);
    dispImageThree.cropFrom(dispImage,vWidth/2,0,vWidth/2,vHeight);

    dispImageTwo.resize(crpX,crpY);
    dispImageThree.resize(crpX,crpY);
    //dispImage.draw(0,0);

    inImage.setFromPixels(dispImageTwo.getPixels());
    inImage.update();

    inImageTwo.setFromPixels(dispImageThree.getPixels());
    inImageTwo.update();

    model.run_image_to_image(inImage,outImage, input_range, output_range);
    modelTwo.run_image_to_image(inImageTwo,outImageTwo, input_range, output_range);

    drawImage.setFromPixels(outImage.getPixels());
    drawImage.resize(vWidth/2,vHeight);
    drawImage.draw(dispXOff,dispYOff);

    drawImageTwo.setFromPixels(outImageTwo.getPixels());
    drawImageTwo.resize(vWidth/2,vHeight);
    drawImageTwo.draw(dispXOff+(vWidth/2),dispYOff);

}

void ofApp::drawFlockingPolylines()
{
    size_t b_cnt = 0;
    for(size_t i = 0; i<rpolyLines.size(); i++) {
        //calculate distance between centroid and boid, store vetexes
        vector<glm::vec3> rpLineVerts = rpolyLines[i].getVertices();
        glm::vec3 center = rpolyLines[i].getCentroid2D();
        //clear this polyline and add new verts
        rpolyLines[i].clear();
        float xOffset = flock.boids[b_cnt].lastLoc.x - center.x;
        float yOffset = flock.boids[b_cnt].lastLoc.y - center.y;
        for(auto vrtx: rpLineVerts) {
            rpolyLines[i].addVertex(vrtx.x+xOffset,vrtx.y+yOffset);
        }
        drawReSampledPolylines(rpolyLines[i],0,0);
        b_cnt++;
    }
}
void ofApp::drawReSampledPolylines(ofPolyline &resampledPoly, int tx, int ty)
{
  //  ofPushMatrix();
    //ofTranslate(-dispXOff,-dispYOff);
   // ofTranslate(tx-dispXOff*2,ty-dispYOff);
    //ofDrawRectangle(0, 0, ofGetWidth() / 2, ofGetHeight());
    
    
    // Draw the resampled polyline in yellow.
    //ofSetColor(ofColor::yellow);
    resampledPoly.draw();
   // ofPopMatrix();
    // Draw its vertices.
   // ofSetColor(255, 255, 127);
   // for (auto vertex: resampledPoly.getVertices())
   // {
   //     ofDrawCircle(vertex, 3);
   // }
    
   // ofSetColor(255);
   // ofDrawBitmapString("Resampled by Spacing", 14, ofGetHeight() - 14);
    ofPushMatrix();
    float time = ofGetElapsedTimef();
    size_t stride = (size_t)ofRandom(1,1);
    for (std::size_t i = 0; i < resampledPoly.size(); i=i+stride)
    {
        float phase = ofMap(i, 0, resampledPoly.size(), 0, glm::pi<float>());
        float scaling = ofRandom(25,50) * sin(time + phase);
        
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
        ofDrawCircle(positiveNormalVertexOffset, scaling / 8);
        ofDrawLine(vertex, positiveNormalVertexOffset);
        
        //ofSetColor(255, 0, 0, 80);
        ofDrawCircle(negativeNormalVertexOffset, scaling / 8);
        ofDrawLine(vertex, negativeNormalVertexOffset);
    }
    ofPopMatrix();
    //ofTranslate(dispXOff,dispYOff);
}

void ofApp::setupGUI()
{

    freeDrawGui.setup();
    freeDrawGui.setPosition(50,50);
    freeDraw.setName("Free Draw");
    freeDraw.addListener(this, &ofApp::ToggleFreeDrawMode);
    freeDrawGui.add(freeDraw.set("Free Draw"));
    freeDrawGui.add(fdLineWidth.set("Line Width", 3.0, 1.0, 6.0));
    closePolyline.setName("Close Polyline");
    freeDrawGui.add(closePolyline.set("Close Polyline"));

    flockDrawGui.setup();
    flockDrawGui.setPosition(freeDrawGui.getWidth(), 50);
    flockDraw.setName("Flock Draw");
    flockDraw.addListener(this, &ofApp::ToggleFlockDrawMode);
    flockDrawGui.add(flockDraw.set("Flock Draw"));
    flockDrawGui.add(separateDistance.set("Separation",25,1,250));
    flockDrawGui.add(alignDistance.set("Align",25,1,250));
    flockDrawGui.add(cohesionDistance.set("Cohesion",25,1,250));
    flockDrawGui.add(maxSpeed.set("Max Speed",1.5,0.5,25.0));
   // drawTestImage.addListener(this, &ofApp::ToggleDrawTestImage);
    //drawTestImage.setName("TestImage");
   // rangeGui.add(drawTestImage.set("TestImage"));
    //rangeGui.add(scaleRange.set("Scale Range",25,25,100));
    //rangeGui.add(radiusRange.set("radius range",1,1,5));

}

void ofApp::ToggleFreeDrawMode(bool &pressed)
{
    if(pressed) {
        //make sure flockDraw is off
        flockDraw = false;
    }
}

void ofApp::ToggleFlockDrawMode(bool &pressed)
{
    if(pressed) {
        //make sure flockDraw is off
        freeDraw = false;
    }
}


void ofApp::ToggleDrawTestImage(bool &pressed)
{
    printf("IN PRESS: %d %d\n",pressed,drawTest);
    if(drawTest) {
        drawTest = false;
    } else {
        drawTest = true;
    }
    //drawTest = drawTest ? false : true;
}

void ofApp::drawGUI()
{
   freeDrawGui.draw();
   flockDrawGui.draw();
}

string ofApp::getImageFileName(int cnt)
{
    string fname;
    char prefix[4];
    sprintf(prefix,"%04d",cnt);
    fname.append("images/");
    fname.append(prefix);
    fname.append(".jpg.png");

    return fname;
}
//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    switch(key) {
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
        load_model_index(key-'1');
        break;


    case 'f':
        if(rsampPline) {
            rsampPline = false;
        } else {
            rsampPline = true;
        }
        break;
    case 'c':
        if(closePline) {
            closePline = false;
        } else {
            closePline = true;
        }
        break;
    case 'n':
        polyLines.clear();
        rpolyLines.clear();
        flock.boids.clear();
        break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    if(freeDraw || flockDraw) {
        polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
    }
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

    if(freeDraw || flockDraw) {
        ofPolyline pLine;
        polyLines.push_back(pLine);
        polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
    }
    if(flockDraw) {
        //polyline for resampling
        ofPolyline rpLine;
        rpolyLines.push_back(rpLine);
    }
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

    if(freeDraw) {
        polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
        if(closePolyline)
            polyLines[polyLines.size()-1].close();
    }
    if(flockDraw) {
        polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
        //must close polyline
        polyLines[polyLines.size()-1].close();
        rpolyLines[rpolyLines.size()-1] = polyLines[polyLines.size()-1].getResampledBySpacing(20);
        //add a boid at centroid of closed resampled pline
        flock.addBoid(rpolyLines[rpolyLines.size()-1].getCentroid2D().x,rpolyLines[rpolyLines.size()-1].getCentroid2D().y);
        //pop_back polyline
        polyLines.pop_back();
    }
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
