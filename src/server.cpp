
#ifdef __unix__
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h> // 包含 inet_pton 函数
#endif

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#pragma comment(lib, "ws2_32.lib")
#endif


struct MyStruct {
 uint32_t custom_mode; /*<  A bitfield for use for autopilot-specific flags*/
 uint8_t type; /*<  Vehicle or component type. For a flight controller component the vehicle type (quadrotor, helicopter, etc.). For other components the component type (e.g. camera, gimbal, etc.). This should be used in preference to component id for identifying the component type.*/
 uint8_t autopilot; /*<  Autopilot type / class. Use MAV_AUTOPILOT_INVALID for components that are not flight controllers.*/
 uint8_t base_mode; /*<  System mode bitmap.*/
 uint8_t system_status; /*<  System status flag.*/
 uint8_t mavlink_version; /*<  MAVLink version, not writable by user, gets added by protocol because of magic data type: uint8_t_mavlink_version*/
} ;



#ifdef __linux__

int InitServer()
{
    int server_socket=socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = INADDR_ANY;
	server_addr.sin_port = htons(5760);


    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr))<0) {
        printf("bind error\n");
		return -1;
	}
    printf("server bind complete!\n");


    if (listen(server_socket, 20) == -1) 
	{
		printf("Listen Failed!\n");
		
		return -1;
	}


    printf("server listen complete!\n");

	return server_socket;
}
#endif  //__linux__

#ifdef WIN32
int InitServer()
{
	WSADATA wsaData;
	int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (result != 0) {
		std::cerr << "WSAStartup failed with error: " << result << std::endl;
		return 1;
	}

	// 创建套接字
	SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (listenSocket == INVALID_SOCKET) {
		std::cerr << "Failed to create socket with error: " << WSAGetLastError() << std::endl;
		WSACleanup();
		return 1;
	}

	// 绑定套接字到指定端口
	sockaddr_in serverAddr;
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_addr.s_addr = INADDR_ANY;
	serverAddr.sin_port = htons(5760); // 使用端口8080

	if (bind(listenSocket, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
		std::cerr << "Bind failed with error: " << WSAGetLastError() << std::endl;
		closesocket(listenSocket);
		WSACleanup();
		return 1;
	}

	if (listen(listenSocket, SOMAXCONN) == SOCKET_ERROR) {
		std::cerr << "Listen failed with error: " << WSAGetLastError() << std::endl;
		closesocket(listenSocket);
		WSACleanup();
		return 1;
	}

	return listenSocket;
}

#endif // WIN32



int Connect2Client(int server_socket)
{
	printf("==============Connect2Client==============\n");
	struct sockaddr_in remote_addr;
	socklen_t remote_addr_size = sizeof(remote_addr);
	int new_server_socket = accept(server_socket, (struct sockaddr*)&remote_addr, &remote_addr_size);
	if (new_server_socket < 0)
	{
		printf("<Connect2Client>Server Accept Failed!\n");
		return -1;
	}
	return new_server_socket;
}




int recvStructInfo(int new_client_socket, MyStruct & mystruct) {

	int total = sizeof(MyStruct);

	int sum = recv(new_client_socket, (char *) & mystruct, total, 0);

	if (sum < 0) {
		printf("<recvStructInfo> Client Recieve Data Failed!\n");
		return 0;
	}

	while (sum != total) {
		char *temp_data = new char[total - sum];
		int length = recv(new_client_socket, temp_data, total - sum, 0);
		if (length > 0) {
			memcpy(&mystruct + sum, temp_data, length);
			sum += length;
		}
		delete temp_data;
		if (length <= 0) {
			printf("<recvStructInfo> Client Recieve Data Failed!\n");
			return 0;
		}
	}

	printf("type = %u, autopilot = %u, base_mode = %u, custom_mode = %u, system_status = %u, mavlink_version = %u\n",
		mystruct.type, mystruct.autopilot, mystruct.base_mode, mystruct.custom_mode, mystruct.system_status, mystruct.mavlink_version);


	return 1;
}






int main() {


	auto server_socket = InitServer();

	printf("Server Start Listening ...\n");

	int new_server_socket = Connect2Client(server_socket);


	while (true)
	{


		MyStruct mystruct;
		if (!(recvStructInfo(new_server_socket, mystruct)))
		{	
			
			new_server_socket = Connect2Client(server_socket);

			continue;
		}


	}

}
