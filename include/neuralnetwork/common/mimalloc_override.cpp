#if defined(MYODDWEB_USE_MIMALLOC)

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4559) // 'operator new': redefinition; the function gains __declspec(restrict)
#endif

#include "../libraries/mimalloc/include/mimalloc-new-delete.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif
