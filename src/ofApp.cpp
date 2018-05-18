#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    //set video w/h
    ofBackground(30);
    vWidth = 768;
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

    dispImage.allocate(crpX,crpY,OF_IMAGE_COLOR);
    outImage.allocate(crpX,crpY,OF_IMAGE_COLOR);
    inImage.allocate(crpX,crpY,OF_IMAGE_COLOR);

    drawImage.allocate(vWidth,vHeight,OF_IMAGE_COLOR);
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

    setupGUI();
    //for (int i = startCount; i > 0; i--) {
     //       flock.addBoid();
     //   }


    //setup flock


}
void ofApp::load_model_index(int index) {
    cur_model_index = ofClamp(index, 0, models_dir.size()-1);
    load_model(models_dir.getPath(cur_model_index));
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
//--------------------------------------------------------------
void ofApp::draw(){
    if(flock.boids.size() > 0) {
        for(size_t i = 0; i<flock.boids.size(); i++) {
            flock.boids[i].setSeperateDistance(separateDistance);
            flock.boids[i].setCohesionDistance(cohesionDistance);
            flock.boids[i].setAlignDistance(alignDistance);
        }
        flock.update();
    }
    drawGUI();

    drawFBO.begin();
    ofClear(0,0,0);
    drawFBO.end();
    drawFBO.begin();
    //ofPushMatrix();
    //ofSetLineWidth(20);
    //for(auto &pline: polyLines) {
    //    pline.draw();
   // }
    flock.draw();
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


   // ofPopMatrix();
    drawFBO.end();
    drawFBO.draw(dispXOff,dispYOff);
   // ofPopMatrix();


    drawFBO.readToPixels(dispImage.getPixels());
    dispImage.update();
    dispImage.resize(crpX,crpY);
    //dispImage.draw(0,0);

    inImage.setFromPixels(dispImage.getPixels());
    inImage.update();
    model.run_image_to_image(inImage,outImage, input_range, output_range);
    drawImage.setFromPixels(outImage.getPixels());
    drawImage.resize(vWidth,vHeight);
    drawImage.draw(dispXOff+vWidth,dispYOff);


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
    //ofPushMatrix();
    float time = ofGetElapsedTimef();
    size_t stride = (size_t)ofRandom(1,strideRange);
    for (std::size_t i = 0; i < resampledPoly.size(); i=i+stride)
    {
        float phase = ofMap(i, 0, resampledPoly.size(), 0, glm::pi<float>());
        float scaling = ofRandom(25,scaleRange) * sin(time + phase);
        
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
        ofDrawCircle(positiveNormalVertexOffset, scaling / ofRandom(1,radiusRange));
        ofDrawLine(vertex, positiveNormalVertexOffset);
        
        //ofSetColor(255, 0, 0, 80);
        ofDrawCircle(negativeNormalVertexOffset, scaling /ofRandom(1,radiusRange));
        ofDrawLine(vertex, negativeNormalVertexOffset);
    }
  //  ofPopMatrix();
    //ofTranslate(dispXOff,dispYOff);
}

void ofApp::setupGUI()
{
    rangeGui.setup();
    rangeGui.setPosition(50,50);
    rangeGui.add(separateDistance.set("Separation",50,1,500));
    rangeGui.add(alignDistance.set("Align",50,1,500));
    rangeGui.add(cohesionDistance.set("Cohesion",50,1,500));
    //rangeGui.add(scaleRange.set("Scale Range",25,25,100));
    //rangeGui.add(radiusRange.set("radius range",1,1,5));

}

void ofApp::drawGUI()
{
    rangeGui.draw();
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
    //polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
    flock.addBoid(x-dispXOff,y-dispYOff);
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

   // ofPolyline pLine;
   // ofPolyline rpLine;
  //  polyLines.push_back(pLine);
  //  rpolyLines.push_back(rpLine);
  //  polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

  //  polyLines[polyLines.size()-1].addVertex(x-dispXOff,y-dispYOff);
  //  if(closePline)
   //     polyLines[polyLines.size()-1].close();
  //  if(rsampPline) {
  //      rpolyLines[rpolyLines.size()-1] = polyLines[polyLines.size()-1].getResampledBySpacing(20);
  //      flock.addBoid(rpolyLines[rpolyLines.size()-1].getCentroid2D().x,rpolyLines[rpolyLines.size()-1].getCentroid2D().y);
  //  }
    
    
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
