#include <iostream>
#include <pthread.h>

void* printMessage(void* arg) {
    const char* message = static_cast<const char*>(arg);
    std::cout << message << std::endl;
    return nullptr;
}

int main() {
    pthread_t threads[4];

    const char* messages[4] = {
        "Thread 1",
        "Thread 2",
        "Thread 3",
        "Thread 4"
    };

    const char* messages1[4] = {
        "Thread 1 penis",
        "Thread 2",
        "Thread 3",
        "Thread 4"
    };

    for (int i = 0; i < 4; ++i) {
        pthread_create(&threads[i], nullptr, printMessage, (void*)messages1[i]);
    }

    for (int i = 0; i < 4; ++i) {
        pthread_join(threads[i], nullptr);
    }

    std::cout << "All threads finished.\n";
    return 0;
}
