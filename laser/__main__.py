import asyncio

from .main import check_deps, main

if __name__ == "__main__":
    check_deps()
    asyncio.run(main())
