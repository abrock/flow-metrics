#include "mail.h"

namespace KITTI {

Mail::Mail (std::string email, std::string from, std::string subject) {
    if (email.compare("")) {
        char cmd[2048];
        sprintf(cmd,"/usr/lib/sendmail -t -f noreply@cvlibs.net");
        mail = popen(cmd,"w");
        fprintf(mail,"To: %s\n", email.c_str());
        fprintf(mail,"From: %s\n", from.c_str());
        fprintf(mail,"Subject: %s\n", subject.c_str());
        fprintf(mail,"\n\n");
    } else {
        mail = 0;
    }
}

Mail::~Mail() {
    if (mail) {
        pclose(mail);
    }
}

void Mail::msg (const char *format, ...) {
    va_list args;
    va_start(args,format);
    if (mail) {
        vfprintf(mail,format,args);
        fprintf(mail,"\n");
    }
    vprintf(format,args);
    printf("\n");
    va_end(args);
}

void Mail::msg (std::string str) {
    if (mail) {
        fprintf(mail,"%s\n",str.c_str());
    }
    printf("%s\n",str.c_str());
}

void Mail::finalize (bool success,std::string benchmark, std::string result_sha, std::string user_sha) {
    if (success) {
        msg("Your evaluation results are available at:");
        msg("http://www.cvlibs.net/datasets/kitti/user_submit_check_login.php?benchmark=%s&user=%s&result=%s",benchmark.c_str(),user_sha.c_str(), result_sha.c_str());
    } else {
        msg("An error occured while processing your results.");
        msg("Please make sure that the data in your zip archive has the right format!");
    }
}


} // namespace KITTI
