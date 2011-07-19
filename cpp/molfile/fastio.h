/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2006 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: fastio.h,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.19 $       $Date: 2006/01/27 20:31:44 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This is a simple abstraction layer for system-dependent I/O calls
 * that allow plugins to do binary I/O using the fastest possible method.
 *
 * This code is intended for use by binary trajectory reader plugins that
 * work with multi-gigabyte data sets, reading only binary data.
 *
 ***************************************************************************/

#ifndef _FASTIO_H__
#define _FASTIO_H__

/* Compiling on windows */
#if defined(_MSC_VER)

#if 1 
/* use native Windows I/O calls */
#define FASTIO_NATIVEWIN32 1

#include <stdio.h>
#include <windows.h>

typedef HANDLE fio_fd;
typedef LONGLONG fio_size_t;
typedef void * fio_caddr_t;

typedef struct {
  fio_caddr_t iov_base;
  int iov_len;
} fio_iovec;

#define FIO_READ  0x01
#define FIO_WRITE 0x02

#define FIO_SEEK_CUR  FILE_CURRENT
#define FIO_SEEK_SET  FILE_BEGIN
#define FIO_SEEK_END  FILE_END

#else

/* Version for machines with plain old ANSI C  */

#include <stdio.h>

typedef FILE * fio_fd;
typedef size_t fio_size_t;  /* MSVC doesn't uinversally support ssize_t */
typedef void * fio_caddr_t; /* MSVC doesn't universally support caddr_t */

typedef struct {
  fio_caddr_t iov_base;
  int iov_len;
} fio_iovec;

#define FIO_READ  0x01
#define FIO_WRITE 0x02

#define FIO_SEEK_CUR SEEK_CUR
#define FIO_SEEK_SET SEEK_SET
#define FIO_SEEK_END SEEK_END

#endif /* plain ANSI C */

#else 

/* Version for UNIX machines */
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

typedef int fio_fd;
typedef off_t fio_size_t;      /* off_t is 64-bits with LFS builds */

/* enable use of kernel readv() if available */
#if defined(__sun) || defined(__APPLE_CC__) || defined(__linux)
#define USE_KERNEL_READV 1
#endif

typedef void * fio_caddr_t;

#if defined(USE_KERNEL_READV)
#include <sys/uio.h>
typedef struct iovec fio_iovec;
#else

typedef struct {
  fio_caddr_t iov_base;
  int iov_len;
} fio_iovec;
#endif


#define FIO_READ  0x01
#define FIO_WRITE 0x02

#define FIO_SEEK_CUR SEEK_CUR
#define FIO_SEEK_SET SEEK_SET
#define FIO_SEEK_END SEEK_END

#endif


#endif
