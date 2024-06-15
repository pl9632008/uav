#include "SqliteOperator.h"
#include "ini.h"
CSqliteOperator::CSqliteOperator()
{
        pDB = NULL;
}


CSqliteOperator::~CSqliteOperator()
{

    if (pDB)
    {
            sqlite3_close(pDB);
            pDB = NULL;
    }
}


int CSqliteOperator::openDB(const std::string& path)
{
        return sqlite3_open(path.c_str(), &pDB);
}

int CSqliteOperator::doSql(const std::string & sql){

    char *szMsg = NULL;
    return sqlite3_exec(pDB, sql.c_str(), NULL, NULL, &szMsg);
}


int CSqliteOperator::insertResult(const int & taskID,const std::string & imgName, const double & lon, const double & lat, const double & alt, const bool & isValid){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"INSERT INTO result (taskID, imgName, lon, lat, alt, isValid) VALUES ("<<taskID<<",'"<<imgName<<"',"<<std::setprecision(15)<<lon<<","<<std::setprecision(15)<<lat<<","<<std::setprecision(15)<<alt<<","<<isValid<<");";
    sql =  oss.str();
    return doSql(sql);
}

int CSqliteOperator::insertLocation(const int &locImgID, const int & itemID, const std::string & label, const int &type, const std::string &points){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"INSERT INTO location (locImgID, itemID, label, type, points) VALUES ("<<locImgID<<","<<itemID<<",'"<<label<<"',"<<type<<",'"<<points<<"');";
    sql =  oss.str();
    return doSql(sql);
}

int CSqliteOperator::deleteResult(const int & imgID){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"DELETE FROM result WHERE imgID = "<<imgID<<";";
    sql =  oss.str();
    return doSql(sql);
}

int CSqliteOperator::deleteResultTask(const int & taskID){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"DELETE FROM result WHERE taskID = "<<taskID<<";";
    sql =  oss.str();
    return doSql(sql);
}

int CSqliteOperator::deleteOneLocation(const int & locImgID, const int & itemID){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"DELETE FROM location WHERE locImgID = "<<locImgID<<" AND itemID = "<<itemID<<";";
    sql =  oss.str();
    return doSql(sql);
}

int CSqliteOperator::deleteLocation(const int & locImgID){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"DELETE FROM location WHERE locImgID = "<<locImgID<<";";
    sql =  oss.str();
    return doSql(sql);
}

int CSqliteOperator::updateResult(const int & taskID, const int &imgID, const std::string & imgName, const double & lon, const double & lat, const double & alt, const bool & isValid){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"UPDATE result SET taskID = "<<taskID<<", imgID = "<<imgID<<", imgName = '"<<imgName<<"', lon = "<<std::setprecision(15)<<lon<<", lat = "<<std::setprecision(15)<<lat<<", alt = "<<std::setprecision(15)<<alt<< ", isValid = "<<isValid<<" WHERE imgID = "<<imgID<<";";
    sql = oss.str();
    return doSql(sql);
}

int CSqliteOperator::updateLocation(const int &locImgID, const int &itemID, const std::string &label, const int &type, const std::string & points){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"UPDATE location SET locImgID = "<<locImgID<<", itemID = "<<itemID<<", label = '"<<label<<"', type = "<<type<<", points = '"<<points<<"' WHERE locImgID = "<<locImgID<< " AND itemID = "<<itemID<<";";
    sql = oss.str();
    return doSql(sql);
}

 std::vector< std::vector<std::string>> CSqliteOperator::queryResult(const int & taskID){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"SELECT * FROM result WHERE taskID = "<<taskID<<";";
    sql = oss.str();
    auto res = query(sql);
    return res;
}

 std::vector< std::vector<std::string>> CSqliteOperator::queryResult(){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"SELECT * FROM result;";
    sql = oss.str();
    auto res = query(sql);
    return res;
}

 std::vector< std::vector<std::string>> CSqliteOperator::queryLocation(const int & locImgID){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"SELECT * FROM location WHERE locImgID = "<<locImgID<<";";
    sql = oss.str();
    auto res = query(sql);
    return res;
}

 std::vector< std::vector<std::string>> CSqliteOperator::queryLocation(){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"SELECT * FROM location;";
    sql = oss.str();
    auto res = query(sql);
    return res;
}

 std::vector<std::vector<std::string>> CSqliteOperator::queryPageResult(const int &taskId , const int & pageIndex , const int & pageSize)
 {
     std::ostringstream oss;
     std::string sql;
     oss.str("");
     oss << "SELECT * FROM result ";

     int offset = (pageIndex - 1) * pageSize;


     oss << "WHERE taskID = " << taskId;

     oss << " LIMIT "<<pageSize<<" OFFSET "<<offset<<";";
     sql = oss.str();
     auto res = query(sql);
     return res;
 }

 std::vector< std::vector<std::string>> CSqliteOperator::queryPageResult(const std::vector<int> &taskIds , const int & pageIndex , const int & pageSize){
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss << "SELECT * FROM result ";

    int offset = (pageIndex - 1) * pageSize;

    size_t count = taskIds.size();
    if (count > 0) {
        oss << "WHERE ";
    }
    for (size_t i = 0; i < count; ++i) {
        oss << "taskID = " << taskIds[i];
        if (i != count - 1) {
            oss << " OR ";
        }
    }

    oss << " LIMIT "<<pageSize<<" OFFSET "<<offset<<";";
    sql = oss.str();
    auto res = query(sql);
    return res;
}

int CSqliteOperator::queryCntResult(const int& taskID)
{
    std::ostringstream oss;
    std::string sql;
    oss.str("");
    oss<<"SELECT COUNT(*) FROM result ";
    oss << "WHERE taskID = " << taskID;
    sql = oss.str();
    auto res = query(sql);
    return std::stoi(res[0][0]);
}


std::vector< std::vector<std::string>> CSqliteOperator::queryPageLocation(const int & locImgID , const int & pageIndex , const int & pageSize){
   std::ostringstream oss;
   std::string sql;

   int offset = (pageIndex - 1) * pageSize;
   oss.str("");
   oss<<"SELECT * FROM result WHERE locImgID = "<<locImgID<<" LIMIT "<<pageSize<<" OFFSET "<<offset<<";";
   sql = oss.str();
   auto res = query(sql);
   return res;
}

int CSqliteOperator::queryCntLocation(const int& locImgID)
{
   std::ostringstream oss;
   std::string sql;
   oss.str("");
   oss<<"SELECT COUNT(*) FROM location;";
   oss << "WHERE locImgID = " << locImgID;
   sql = oss.str();
   auto res = query(sql);
   return std::stoi(res[0][0]);
}

int CSqliteOperator::queryCntLocation(const std::vector<int>& locImgIDs)
{
   std::ostringstream oss;
   std::string sql;
   oss.str("");
   oss<<"SELECT COUNT(*) FROM location ";
   size_t count = locImgIDs.size();
   if (count > 0) {
       oss << "WHERE ";
   }
   for (size_t i = 0; i < count; ++i) {
       oss << "locImgID = " << locImgIDs[i];
       if (i != count - 1) {
           oss << " OR ";
       }
   }
   sql = oss.str();
   auto res = query(sql);
   if (res.size() == 0 || res[0].size() == 0) {
       return 0;
   }
   return std::stoi(res[0][0]);
}

int CSqliteOperator::queryCntLocationByLabel(const std::vector<int>& locImgIDs, const std::string label)
{
   std::ostringstream oss;
   std::string sql;
   oss.str("");
   oss<<"SELECT COUNT(*) FROM location ";
   size_t count = locImgIDs.size();
   if (count > 0) {
       oss << "WHERE ((";
       for (size_t i = 0; i < count; ++i) {
           oss << "locImgID = " << locImgIDs[i];
           if (i != count - 1) {
               oss << " OR ";
           }
       }
       if (label.size() == 0) {
           return 0;
       }
       oss << ") AND label LIKE '" << label << "')";
   } else {
       return 0;
   }

   sql = oss.str();
   auto res = query(sql);
   if (res.size() == 0 || res[0].size() == 0) {
       return 0;
   }
   return std::stoi(res[0][0]);
}

 std::vector< std::vector<std::string>> CSqliteOperator::query(std::string & sql){
    char *errMsg = 0;
    char **results;
    int rows, columns;
    int rc;

    rc = sqlite3_get_table(pDB, sql.c_str(), &results, &rows, &columns, &errMsg);
    if (rc != SQLITE_OK) {
         sqlite3_free(errMsg);
     }
    std::vector< std::vector<std::string>> res;
    for (int i = 1; i <= rows; i++) {
        std::vector<std::string> temp;
       for (int j = 0; j < columns; j++) {
           std::string str(results[i*columns +j]);
           temp.push_back(str);
       }
       res.push_back(temp);

    }
    return res;

}


int CSqliteOperator::getLastInsertRowid(){
    std::string sql = "select last_insert_rowid();";
    char *errMsg = 0;
    char **results;
    int rows, columns;
    int rc;

    rc = sqlite3_get_table(pDB, sql.c_str(), &results, &rows, &columns, &errMsg);
    if (rc != SQLITE_OK) {
         sqlite3_free(errMsg);
         return -1;
     }
    int res = 0;
    for (int i = 1; i <= rows; i++) {
       for (int j = 0; j < columns; j++) {
           res = atoi(results[i * columns + j]);
       }

    }

    return res;
}


void CSqliteOperator::initSql(bool isDebug){

    if (isDebug) {
        this->openDB("../PLATFORM/tools/sift/database/ai.rwedb");
    } else {
        ini::iniReader config;
        bool ret = config.ReadConfig("../config/config.ini");
        if(!ret){
            printf("initial sql failed!\n");
            return;
        };

        std::string database = config.ReadString("sql", "database", "");
        this->openDB(database);
    }

    this->doSql(R"(
                         PRAGMA FOREIGN_KEYS=ON;
                     )");

    this->doSql(R"(
                        CREATE TABLE IF NOT EXISTS result (
                        taskID INT,
                        imgID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        imgName VARCHAR(50),
                        lon DOUBLE,
                        lat DOUBLE,
                        alt DOUBLE,
                        isValid TINYINT);
                        )"
                     );

    this->doSql(R"(
                        CREATE TABLE IF NOT EXISTS location (
                        locImgID INT,
                        itemID INT,
                        label VARCHAR(20),
                        type TINYINT,
                        points TEXT,
                        CONSTRAINT fk_locImgID FOREIGN KEY (locImgID) REFERENCES result(imgID) ON UPDATE CASCADE ON DELETE CASCADE);
                        )"
                    );
    this->doSql(R"(
                        CREATE INDEX IF NOT EXISTS idx_locImgID ON location(locImgID);
                     )");

}



void CSqliteOperator::initSql(std::string & sql_path){


    this->openDB(sql_path);

    this->doSql(R"(
                         PRAGMA FOREIGN_KEYS=ON;
                     )");

    this->doSql(R"(
                        CREATE TABLE IF NOT EXISTS result (
                        taskID INT,
                        imgID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        imgName VARCHAR(50),
                        lon DOUBLE,
                        lat DOUBLE,
                        alt DOUBLE,
                        isValid TINYINT);
                        )"
                     );

    this->doSql(R"(
                        CREATE TABLE IF NOT EXISTS location (
                        locImgID INT,
                        itemID INT,
                        label VARCHAR(20),
                        type TINYINT,
                        points TEXT,
                        CONSTRAINT fk_locImgID FOREIGN KEY (locImgID) REFERENCES result(imgID) ON UPDATE CASCADE ON DELETE CASCADE);
                        )"
                    );
    this->doSql(R"(
                        CREATE INDEX IF NOT EXISTS idx_locImgID ON location(locImgID);
                     )");

}


void CSqliteOperator::testSql(){

    //return 0 represent succeed;
//    auto a =this->insertResult(66,"seee.jpg",33.4,33.3,10,1);
//    auto b =this->insertLocation(60,25,"wshizcxel",66,"gfhhh");
//    auto c =this->deleteResult(11);
//    auto d =this->deleteOneLocation(12,11);
//    auto e =this->updateResult(100,4,"dxcxzd",1111,6,7,0);
//    auto f =this->updateLocation(12,9,"66111116",0,"7,9,8,6,4");
//    auto g =this->queryResult();
//    auto h =this->queryResult(0);
//    auto i =this->queryResult(100);
//    auto j =this->queryLocation();
//    auto k =this->queryLocation(2);
//    auto l =this->getLastInsertRowid();

}

