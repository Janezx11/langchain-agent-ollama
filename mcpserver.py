from mcp.server.fastmcp import FastMCP
import requests, re, html, logging
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("MyAgentServer")

# --- 公共 session ---
def make_session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3, status_forcelist=(500, 502, 504))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return s

SESSION = make_session()

# --- MCP 工具 ---
@mcp.tool()
def web_search(query: str) -> str:
    """
    用于搜索最新信息（新闻、天气、实时数据、未知知识）。

    使用场景：
    - 用户询问最新信息
    - 不确定答案
    - 需要获取网页链接

    返回：首条结果的标题、摘要和URL。
    """
    try:
        resp = SESSION.get("https://www.bing.com/search", params={"q": query}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        result = soup.find("li", {"class": "b_algo"})
        if not result:
            return "未找到结果"
        h2 = result.find("h2")
        a = h2.find("a") if h2 else None
        title = a.get_text(strip=True) if a else ""
        link = a["href"] if a and a.has_attr("href") else ""
        snippet_tag = result.find("p")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        return f"{title}\n{snippet}\n{link}"
    except Exception as e:
        return f"搜索失败: {e}"

@mcp.tool()
def fetch_page(url: str, deep: bool = False) -> str:
    """
    用于获取网页详细内容。

    使用场景：
    - 已经通过 web_search 获得 URL
    - 需要阅读全文或详细信息

    参数：
    - deep=False：返回摘要（推荐）
    - deep=True：返回全文（较长）

    注意：通常应先调用 web_search，再调用此工具。
    """
    url = url.strip()
    try:
        resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 删除干扰元素
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
            tag.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 优先找 article / main / 正文区域
        content = soup.find("article") or soup.find("main") or soup.find("div", {"role": "main"})
        if content:
            text = content.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        text = html.unescape(text)
        text = re.sub(r"\n{2,}", "\n\n", text).strip()
        if len(text) > 8000:
            text = text[:8000] + "\n\n[已截断]"
        if deep:
            return text
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20][:5]
        return f"{title}\n\n" + "\n\n".join(paras)
    except Exception as e:
        return f"抓取失败: {e}"

@mcp.tool()
def get_weather(city: str, days: int = 1) -> str:
    """
    查询城市天气。

    使用场景：
    - 用户询问天气
    - 比 web_search 更优先使用

    参数：
    - days=1：今天
    - days=2：包含明天
    """
    try:
        resp = SESSION.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        weathers = data.get("weather", [])[:days]
        results = []
        for w in weathers:
            date = w.get("date", "")
            max_temp = w.get("maxtempC", "")
            min_temp = w.get("mintempC", "")
            hourly = w.get("hourly", [])
            mid = hourly[4] if len(hourly) > 4 else (hourly[-1] if hourly else {})
            desc_cn = mid.get("lang_zh", [{}])[0].get("value", "") or mid.get("weatherDesc", [{}])[0].get("value", "")
            results.append(f"{date}: {min_temp}~{max_temp}°C {desc_cn}")
        return "\n".join(results) if results else "未获取到天气数据"
    except Exception as e:
        return f"天气查询失败: {e}"

if __name__ == "__main__":
    mcp.run()
