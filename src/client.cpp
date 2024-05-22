#include "serial/serial.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include "common/mavlink.h"

#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h> // 包含 inet_pton 函数

#include <mutex>


struct MyStruct {
	uint32_t custom_mode; /*<  A bitfield for use for autopilot-specific flags*/
	uint8_t type; /*<  Vehicle or component type. For a flight controller component the vehicle type (quadrotor, helicopter, etc.). For other components the component type (e.g. camera, gimbal, etc.). This should be used in preference to component id for identifying the component type.*/
	uint8_t autopilot; /*<  Autopilot type / class. Use MAV_AUTOPILOT_INVALID for components that are not flight controllers.*/
	uint8_t base_mode; /*<  System mode bitmap.*/
	uint8_t system_status; /*<  System status flag.*/
	uint8_t mavlink_version; /*<  MAVLink version, not writable by user, gets added by protocol because of magic data type: uint8_t_mavlink_version*/
};


struct Quater{
    float w;
    float x;
    float y;
    float z;

};

struct GPSInformation{
    int32_t lat;//degE7
    int32_t lon;//degE7
    int32_t alt;//mm
    int32_t relative_alt;//

};


struct GPSAndQua{
    int32_t lat;//degE7
    int32_t lon;//degE7
    int32_t alt;//mm
    int32_t relative_alt;//
  
    float roll;
    float pitch;
    float yaw;

};


struct EulerAngles {
    double roll, pitch, yaw;
};

Quater qua;
EulerAngles euler;
GPSInformation gpsinfo;

std::mutex mtx_qua;
std::mutex mtx_gps;


mavlink_status_t status;
mavlink_message_t msg_mav;
int chan = MAVLINK_COMM_0;

struct sockaddr_in client_addr;

EulerAngles ToEulerAngles(Quater q) {
    EulerAngles angles;
 
    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);
 
    // pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);
 
    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}


int initClient() {

	int client_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	if (client_socket == -1) {
		printf("client init error!\n");
		return -1;
	}


	memset(&client_addr, 0, sizeof(client_addr));
	client_addr.sin_family = AF_INET;
	client_addr.sin_addr.s_addr = inet_addr("127.0.0.1");//192.168.2.32
	client_addr.sin_port = htons(5760);

	printf("waiting for starting server...\n");

	while (connect(client_socket, (sockaddr*)&client_addr, sizeof(client_addr)) == -1);

	printf("connect server successfully!\n");
	return client_socket;
}

int reconnect() {

	int new_client_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	int res = connect(new_client_socket, (sockaddr*)&client_addr, sizeof(client_addr));

	return new_client_socket;

}




int sendStructInfo(int new_client_socket, MyStruct & mystruct) {

	int ret = send(new_client_socket, (char*)& mystruct, sizeof(MyStruct), MSG_NOSIGNAL);

	return ret;
}



void SerialThread(serial::Serial& ser) {

	auto socket_client = initClient();
	while (true) {
		try {
			uint8_t serialData;
			ser.read(&serialData, 1);

			if (mavlink_parse_char(chan, serialData, &msg_mav, &status)) {
				printf("Received message with ID %d, sequence: %d from component %d of system %d\n", msg_mav.msgid, msg_mav.seq, msg_mav.compid, msg_mav.sysid);

				switch (msg_mav.msgid) {
				case MAVLINK_MSG_ID_HEARTBEAT: // ID for GLOBAL_POSITION_INT
				{

					mavlink_heartbeat_t heart_beat;

					// Get all fields in payload (into global_position)
					mavlink_msg_heartbeat_decode(&msg_mav, &heart_beat);
					printf("type = %u, autopilot = %u, base_mode = %u, custom_mode = %u, system_status = %u, mavlink_version = %u\n",
						heart_beat.type, heart_beat.autopilot, heart_beat.base_mode, heart_beat.custom_mode, heart_beat.system_status, heart_beat.mavlink_version);

					MyStruct mystruct;
					memcpy(&mystruct, &heart_beat, sizeof(heart_beat));


					int ret = sendStructInfo(socket_client, mystruct);
					if (ret == -1) {

						socket_client = reconnect();

						printf("reconnect socket, drop current message\n");

					}

				}
				break;
				case MAVLINK_MSG_ID_TIMESYNC:
				{
					// Get just one field from payload
					mavlink_timesync_t time_sync;
					mavlink_msg_timesync_decode(&msg_mav, &time_sync);
					printf("tc1 = %lld, ts1 = %lld, target_system = %u, target_component = %u\n",
						time_sync.tc1, time_sync.ts1, time_sync.target_system, time_sync.target_component);

				}
				break;

				case MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
				{
					// Get just one field from payload
					mavlink_global_position_int_t global_position_int;
					mavlink_msg_global_position_int_decode(&msg_mav, &global_position_int);
					printf("time_boot_ms = %u, lat = %d, lon = %d, alt = %d, relative_alt = %d, vx = %d, vy = %d, vz = %d, hdg = %u\n",
						global_position_int.time_boot_ms, global_position_int.lat, global_position_int.lon, global_position_int.alt, global_position_int.relative_alt,
						global_position_int.vx, global_position_int.vy, global_position_int.vz, global_position_int.hdg
						);

					
					{
						std::lock_guard<std::mutex> lock(mtx_gps);
						gpsinfo.alt = global_position_int.alt;
						gpsinfo.lat = global_position_int.lat;
						gpsinfo.lon = global_position_int.lon;
						gpsinfo.relative_alt = global_position_int.relative_alt;
						
					}

				}
				break;

				case MAVLINK_MSG_ID_GIMBAL_DEVICE_ATTITUDE_STATUS:
				{
					mavlink_gimbal_device_attitude_status_t gimbal_device_attitude_status;
					auto quater = gimbal_device_attitude_status.q;
					float w = quater[0];
					float x = quater[1];
					float y = quater[2];
					float z = quater[3];

				
					printf("w = %f, x = %f, y = %f, z = %f\n", w, x, y, z);
					{
						std::lock_guard<std::mutex> lock(mtx_qua);
						qua.w = w;
						qua.x = x;
						qua.y = y;
						qua.z = z;
						
						euler = ToEulerAngles(qua);
						
								
					}


					
				}
				break;

				default:
					break;
				}
			}

		}
		catch (std::exception & e) {
			std::cerr << "Unhandled Exception: " << e.what() << std::endl;
			break;
		}
	}
}



int main() {


	serial::Serial ser;
	try {

		//ser.setPort("COM4");//win
		ser.setPort("/dev/ttyUSB0");//linux
		ser.setBaudrate(57600);
		serial::Timeout timeout = serial::Timeout::simpleTimeout(3000);
		ser.setTimeout(timeout);
		ser.open();


	}
	catch (std::exception & e) {
		std::cerr << "Unhandled Exception: " << e.what() << std::endl;

	}

	std::thread serial_thread(SerialThread, std::ref(ser));
	serial_thread.join();

}
