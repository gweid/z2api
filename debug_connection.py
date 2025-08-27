#!/usr/bin/env python3
"""
连接调试脚本 - 用于诊断Z.AI API连接问题
"""
import asyncio
import logging
import sys
from datetime import datetime
import httpx
from config import settings
from cookie_manager import cookie_manager

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_basic_connection():
    """测试基本网络连接"""
    logger.info("=== 测试基本网络连接 ===")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 测试DNS解析
            logger.info("测试DNS解析...")
            response = await client.get("https://chat.z.ai", timeout=10.0)
            logger.info(f"DNS解析成功，状态码: {response.status_code}")

            # 测试API端点连通性
            logger.info("测试API端点连通性...")
            response = await client.get(
                "https://chat.z.ai/api/chat/completions", timeout=10.0
            )
            logger.info(f"API端点连通性测试，状态码: {response.status_code}")

    except Exception as e:
        logger.error(f"基本网络连接测试失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False

    return True


async def test_cookie_availability():
    """测试Cookie可用性"""
    logger.info("=== 测试Cookie可用性 ===")

    if not settings.COOKIES:
        logger.error("没有配置任何Cookie")
        return False

    logger.info(f"配置了 {len(settings.COOKIES)} 个Cookie")

    for i, cookie in enumerate(settings.COOKIES):
        logger.info(f"Cookie {i+1}: {cookie[:20]}...")

    return True


async def test_cookie_health():
    """测试Cookie健康状态"""
    logger.info("=== 测试Cookie健康状态 ===")

    if not settings.COOKIES:
        logger.error("没有配置任何Cookie")
        return False

    healthy_count = 0
    for i, cookie in enumerate(settings.COOKIES):
        logger.info(f"测试Cookie {i+1}...")
        try:
            is_healthy = await cookie_manager.health_check(cookie)
            if is_healthy:
                logger.info(f"Cookie {i+1} 健康")
                healthy_count += 1
            else:
                logger.warning(f"Cookie {i+1} 不健康")
        except Exception as e:
            logger.error(f"Cookie {i+1} 健康检查异常: {e}")

    logger.info(f"健康Cookie数量: {healthy_count}/{len(settings.COOKIES)}")
    return healthy_count > 0


async def test_api_request():
    """测试API请求"""
    logger.info("=== 测试API请求 ===")

    cookie = await cookie_manager.get_next_cookie()
    if not cookie:
        logger.error("没有可用的Cookie")
        return False

    logger.info(f"使用Cookie: {cookie[:20]}...")

    try:
        import uuid

        test_payload = {
            "stream": True,
            "model": "0727-360B-API",
            "messages": [{"role": "user", "content": "Hello"}],
            "background_tasks": {"title_generation": False, "tags_generation": False},
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": [],
            "model_item": {
                "id": "0727-360B-API",
                "name": "GLM-4.5",
                "owned_by": "openai",
            },
            "params": {},
            "tool_servers": [],
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cookie}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/c/069723d5-060b-404f-992c-4705f1554c4c",
        }

        logger.info("发送API请求...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.UPSTREAM_URL, json=test_payload, headers=headers
            )

            logger.info(f"API请求状态码: {response.status_code}")

            if response.status_code == 200:
                logger.info("API请求成功")
                return True
            else:
                logger.error(f"API请求失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return False

    except httpx.RequestError as e:
        logger.error(f"API请求异常: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False
    except Exception as e:
        logger.error(f"API请求未知异常: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False


async def main():
    """主函数"""
    logger.info("开始连接诊断...")
    logger.info(f"当前时间: {datetime.now()}")
    logger.info(f"上游URL: {settings.UPSTREAM_URL}")
    logger.info(f"上游模型: {settings.UPSTREAM_MODEL}")

    results = []

    # 测试基本连接
    results.append(("基本网络连接", await test_basic_connection()))

    # 测试Cookie可用性
    results.append(("Cookie可用性", await test_cookie_availability()))

    # 测试Cookie健康状态
    results.append(("Cookie健康状态", await test_cookie_health()))

    # 测试API请求
    results.append(("API请求", await test_api_request()))

    # 输出总结
    logger.info("=== 诊断总结 ===")
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")

    # 如果所有测试都通过，给出建议
    if all(result for result in results):
        logger.info("🎉 所有测试通过，连接正常！")
    else:
        logger.info("⚠️  部分测试失败，请检查以下方面：")
        logger.info("1. 网络连接是否正常")
        logger.info("2. Cookie是否有效")
        logger.info("3. Z.AI服务是否可用")
        logger.info("4. 防火墙或代理设置")


if __name__ == "__main__":
    asyncio.run(main())
