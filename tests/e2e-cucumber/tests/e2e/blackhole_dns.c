// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Linux-only LD_PRELOAD fixture for one short-lived CLI child. Redirect DNS and
// connections to a loopback port with a full accept queue: Linux drops new SYNs.
// The real connect syscall/poll still runs, so this tests the client's
// timeout rather than injecting a synthetic timeout error. No host DNS/firewall
// changes, external network, privileged ports, or production test hooks.
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netdb.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

static struct sockaddr_in address = {
    .sin_family = AF_INET,
};
static int listener = -1;
static int filler = -1;

static void require(int result) {
    if (result < 0) {
        perror("blackhole fixture");
        _exit(125);
    }
}

__attribute__((constructor)) static void start_blackhole(void) {
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t length = sizeof(address);
    require(listener = socket(AF_INET, SOCK_STREAM, 0));
    require(bind(listener, (struct sockaddr *)&address, length));
    require(listen(listener, 0));
    require(getsockname(listener, (struct sockaddr *)&address, &length));
    require(filler = socket(AF_INET, SOCK_STREAM, 0));
    require(syscall(SYS_connect, filler, (struct sockaddr *)&address, length));
    // Fail boundedly even if the client regresses to its 30s/600s connect wait.
    alarm(20);
}

int getaddrinfo(const char *node, const char *service,
                const struct addrinfo *hints, struct addrinfo **result) {
    (void)node;
    int (*resolve)(const char *, const char *, const struct addrinfo *,
                   struct addrinfo **) = dlsym(RTLD_NEXT, "getaddrinfo");
    if (!resolve) {
        _exit(125);
    }
    return resolve("127.0.0.1", service, hints, result);
}

int connect(int fd, const struct sockaddr *peer, socklen_t length) {
    (void)peer;
    (void)length;
    return syscall(SYS_connect, fd, (struct sockaddr *)&address, sizeof(address));
}

__attribute__((destructor)) static void stop_blackhole(void) {
    close(filler);
    close(listener);
}
