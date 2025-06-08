#include <ros/ros.h>
#include <iostream>

#include "example.hpp"

int main(int argc, char * argv[])
{
    ros::init(argc, argv, "exmaple_node");
    
    std::cout << "Write new node here!" << std::endl;

    ros::spin();
    ros::shutdown();
    return 0;
}
