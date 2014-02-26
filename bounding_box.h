#pragma once
#include <pyarr.h>

/*
  Top,Left,Bottom,Right
 */
class BoundingBox {
 public:
    BoundingBox() {x1_ = -1; x2_ = -1; y1_ = -1; y2_ = -1; update_metadata();}

    BoundingBox(int x1, int y1, int x2, int y2) 	
     {
	x1_ = x1;
	y1_ = y1;
	x2_ = x2;
	y2_ = y2;
	update_metadata();
     }

    void merge(BoundingBox b2) 
    {
	x1_ = min(x1_, b2.x1_);
	y1_ = min(y1_, b2.y1_);
	x2_ = max(x2_, b2.x2_);
	y2_ = max(y2_, b2.y2_);
	update_metadata();
    }

    int merged_area(BoundingBox b2) const
    {
	BoundingBox tmp(x1_, y1_, x2_, y2_);
	tmp.merge(b2);
	return tmp.area;
    }

    void print() 
    {
	printf("BoundingBox (x1,y1), (x2, y2): (%d, %d), (%d, %d). Valid: %d\n", 
	       x1_, y1_, x2_, y2_, valid_);
    }

 private:
    void update_metadata()
    {
	width = x2_ - x1_;
	height = y2_ - y1_;
	area = width * height;
	valid_ = ((x1_ >= 0) && (y1_ >= 0) &&
		  (x2_ >= x1_) && (y2_ >= y1_));
    }

 public:
    int x1_, y1_, x2_, y2_,
	width, height, area;
    
    bool valid_;
};

// in batch,
// this is much slower than gtkutils.img_util.bounding_boxes_from_seg()
BoundingBox bounding_box_of_label(pyarr<int> matrix, int label) {
    int bbox_y1 = -1;
    int bbox_x1 = -1;
    int bbox_y2 = -1;
    int bbox_x2 = -1;

    const int& height = matrix.dims[0];
    const int& width = matrix.dims[1];

    bool found = false;
    //scan down for top boundary
    for (int row =0; row < height; row++) {
	for (int col = 0; col < width; col++) {
	    if (matrix[ind(row, col)] == label) {
		bbox_y1 = row;
		found = true;
		break;
	    }
	}
	if (found) break;
    }

    found = false;
    //scan up for bottom boundary
    for (int row = height - 1; row >= 0; row--) {
	for (int col = 0; col < width; col++) {
	    if (matrix[ind(row, col)] == label) {
		bbox_y2 = row;
		found = true;
		break;
	    }
	}
	if (found) break;
    }

    found = false;
    //scan right for left boundary
    for (int col = 0; col < width; col++) {
	for (int row = 0; row < height; row++) {
	    if (matrix[ind(row, col)] == label) {
		bbox_x1 = col;
		found = true;
		break;
	    }
	}
	if (found) break;
    }

    found = false;
    //scan left for right boundary
    for (int col = width - 1; col >= 0; col--) {
	for (int row = 0; row < height; row++) {
	    if (matrix[ind(row, col)] == label) {
		bbox_x2 = col;
		found = true;
		break;
	    }
	}
	if (found) break;
    }

    return BoundingBox(bbox_x1, bbox_y1, bbox_x2, bbox_y2);
}
