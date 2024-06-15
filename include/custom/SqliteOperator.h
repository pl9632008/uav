#ifndef SQLITEOPERATOR_H
#define SQLITEOPERATOR_H
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "sqlite3.h"

class CSqliteOperator
{
public:
        CSqliteOperator();
        ~CSqliteOperator();

        int insertResult(const int & taskID, const std::string & imgName, const double & lon, const double & lat, const double & alt, const bool & isValid);
        int insertLocation(const int &locImgID, const int & itemID, const std::string & label, const int &type, const std::string &points);
        int deleteResult(const int & imgID);
        int deleteResultTask(const int & taskID);

        int deleteOneLocation(const int & locImgID, const int & itemID);
        int deleteLocation(const int & locImgID);
        int updateResult(const int & taskID, const int &imgID, const std::string & imgName, const double & lon, const double & lat, const double & alt, const bool & isValid);
        int updateLocation(const int &locImgID, const int &itemID, const std::string &label, const int &type, const std::string & points);
        std::vector< std::vector<std::string>> queryResult(const int & taskID);
        std::vector< std::vector<std::string>> queryResult();
        std::vector< std::vector<std::string>> queryLocation(const int & locImgID);
        std::vector< std::vector<std::string>> queryLocation();
        std::vector< std::vector<std::string>> queryPageResult(const int &taskId , const int & pageIndex , const int & pageSize);
        std::vector< std::vector<std::string>> queryPageResult(const std::vector<int> &taskIds , const int & pageIndex , const int & pageSize);
        std::vector< std::vector<std::string>> queryPageLocation(const int & locImgID , const int & pageIndex , const int & pageSize);
        std::vector< std::vector<std::string>> query(std::string & sql);
        int queryCntResult(const int& taskID);
        int queryCntLocation(const int& locImgID);
        int queryCntLocation(const std::vector<int>& locImgIDs);
        int queryCntLocationByLabel(const std::vector<int>& locImgIDs, const std::string label);

        int openDB(const std::string& path);
        int doSql(const std::string & sql);
        int getLastInsertRowid();
        void testSql();
        void initSql(bool isDebug = false);
        void initSql(std::string & sql_path);

        sqlite3 *pDB;
};

#endif // SQLITEOPERATOR_H
