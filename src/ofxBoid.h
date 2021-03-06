/*
 *  Boid.h
 *  boid
 *
 *  Created by Jeffrey Crouse on 3/29/10.
 *  Copyright 2010 Eyebeam. All rights reserved.
 *  Updated by Takara Hokao
 *	Updated by Andrew Monks
 *
 */

#ifndef BOID_H
#define BOID_H

#include "ofMain.h"
//#include "ofxVectorMath.h"

class Boid {
public:
	Boid();
	Boid(int x, int y);
	
	void update(vector<Boid> &boids);
    void draw();
	
    void seek(ofVec2f target);
    void avoid(ofVec2f target);
    void arrive(ofVec2f target);
	
    void flock(vector<Boid> &boids);
    bool isHit(int x,int y, int radius);
    
	ofVec2f steer(ofVec2f target, bool slowdown);
	
	ofVec2f separate(vector<Boid> &boids);
	ofVec2f align(vector<Boid> &boids);
	ofVec2f cohesion(vector<Boid> &boids);
	
    ofVec2f loc,vel,acc, lastLoc;
    
	float r;
	float maxforce;
	float maxspeed;

    float dispXOff = 192;
    float dispYOff = 216;
    float vWidth = 1536;
    float vHeight = 768;

    float cohesionDistance,alignDistance,seperateDistance;

    inline void setCohesionDistance(float cd) {cohesionDistance = cd;}
    inline void setAlignDistance(float ad) {alignDistance = ad;}
    inline void setSeperateDistance(float sd) {seperateDistance = sd;}
    inline void setMaxSpeed(float ms) { maxspeed = ms;}

};

#endif
