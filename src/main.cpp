#include "rotate.h"
int main(){

    std::shared_ptr<Rotation> rot = std::make_shared<Rotation>();

    rot->initGroundSky();
    rot->runSky();
}
