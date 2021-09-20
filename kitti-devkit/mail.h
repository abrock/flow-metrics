#ifndef MAIL_H
#define MAIL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <string>

namespace KITTI {

class Mail {

public:

    Mail (
            std::string email = "",
            std::string from = "noreply@cvlibs.net",
            std::string subject = "KITTI Evaluation Benchmark");

    ~Mail();

    void msg (const char *format, ...);

    void msg (std::string str);

    void finalize (bool success,std::string benchmark,std::string result_sha="",std::string user_sha="");
    
private:

    FILE *mail;

};

} // namespace KITTI

#endif // MAIL_H
