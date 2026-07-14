import os
import re
import io
from datetime import datetime
from collections import Counter

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image

# ────────────────────────────────────────────────
# 기본 설정
# ────────────────────────────────────────────────
st.set_page_config(page_title="유튜브 댓글 분석기", page_icon="📊", layout="wide")

DEFAULT_STOPWORDS = set("""
그리고 그러나 그래서 하지만 정말 진짜 너무 정도 이거 저거 그거
우리 저희 그냥 근데 이런 저런 그런 있다 없다 하다 되다 이다
것 수 등 및 을 를 은 는 이 가 의 에 에서 으로 로 와 과 도 만
까지 부터 처럼 보다 이나 나 다 요 죠 네요 습니다 입니다
너 나 제 내 좀 더 왜 뭐 진짜로 완전 그냥요
""".split())

FONT_URLS = [
    "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
    "https://cdn.jsdelivr.net/gh/fonts-archive/NanumGothic/NanumGothic.ttf",
]

# ────────────────────────────────────────────────
# 유틸 함수
# ────────────────────────────────────────────────
def get_api_key() -> str:
    key = None
    try:
        key = st.secrets.get("YOUTUBE_API_KEY", None)
    except Exception:
        pass
    if not key:
        key = os.environ.get("YOUTUBE_API_KEY")
    return key


def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    return ""


@st.cache_resource(show_spinner=False)
def get_korean_font_path() -> str:
    font_path = "/tmp/NanumGothic.ttf"
    if os.path.exists(font_path) and os.path.getsize(font_path) > 100000:
        return font_path
    for url in FONT_URLS:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(resp.content)
            if os.path.getsize(font_path) > 100000:
                return font_path
        except Exception:
            continue
    return ""  # 폰트 다운로드 실패 시 기본 폰트 사용


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_video_info(video_id: str, api_key: str) -> dict:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet,statistics", "id": video_id, "key": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return {}
    snippet = items[0]["snippet"]
    stats = items[0]["statistics"]
    return {
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "published": snippet.get("publishedAt", ""),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_comments(video_id: str, api_key: str, max_comments: int, order: str) -> pd.DataFrame:
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": api_key,
            "maxResults": 100,
            "order": order,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            reason = r.json().get("error", {}).get("errors", [{}])[0].get("reason", "")
            raise RuntimeError(f"API 오류 ({r.status_code}): {reason}")

        data = r.json()
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textOriginal", ""),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt", ""),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
            })
            if len(comments) >= max_comments:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    df = pd.DataFrame(comments)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
        df["text_length"] = df["text"].str.len()
    return df


def tokenize(text: str, stopwords: set) -> list:
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    korean_words = re.findall(r"[가-힣]{2,}", text)
    english_words = re.findall(r"[a-zA-Z]{3,}", text)
    words = korean_words + [w.lower() for w in english_words]
    return [w for w in words if w not in stopwords]


def make_circle_mask(size: int = 900) -> np.ndarray:
    x, y = np.ogrid[:size, :size]
    center = size / 2
    mask = (x - center) ** 2 + (y - center) ** 2 > center ** 2
    return 255 * mask.astype(int)


# ────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")

video_input = st.sidebar.text_input("유튜브 영상 URL 또는 ID", placeholder="https://www.youtube.com/watch?v=...")
max_comments = st.sidebar.slider("가져올 댓글 수", 50, 2000, 300, step=50)
order = st.sidebar.selectbox("댓글 정렬", ["relevance", "time"], format_func=lambda x: "관련도순" if x == "relevance" else "최신순")
colormap = st.sidebar.selectbox(
    "워드클라우드 색상",
    ["plasma", "viridis", "magma", "inferno", "cool", "autumn", "Reds", "Blues", "spring"],
)
extra_stopwords_input = st.sidebar.text_area("제외할 단어 (쉼표로 구분)", "")
run = st.sidebar.button("🔍 분석 시작", use_container_width=True, type="primary")

st.title("📊 유튜브 댓글 분석기")
st.caption("영상 URL을 입력하고 분석을 시작하세요.")

# ────────────────────────────────────────────────
# 메인 로직
# ────────────────────────────────────────────────
if run:
    api_key = get_api_key()
    if not api_key:
        st.error("YOUTUBE_API_KEY가 설정되어 있지 않습니다. Streamlit Cloud의 Secrets에 등록해주세요.")
        st.stop()

    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("유효한 유튜브 URL 또는 영상 ID를 입력해주세요.")
        st.stop()

    try:
        with st.spinner("영상 정보를 가져오는 중..."):
            info = fetch_video_info(video_id, api_key)
        if not info:
            st.error("영상을 찾을 수 없습니다.")
            st.stop()

        with st.spinner("댓글을 가져오는 중... (댓글 수가 많으면 시간이 걸릴 수 있어요)"):
            df = fetch_comments(video_id, api_key, max_comments, order)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.stop()

    if df.empty:
        st.warning("댓글이 없거나 댓글이 비활성화된 영상입니다.")
        st.stop()

    # ── 영상 정보 카드 ──
    col1, col2 = st.columns([1, 3])
    with col1:
        if info.get("thumbnail"):
            st.image(info["thumbnail"], use_container_width=True)
    with col2:
        st.subheader(info["title"])
        st.write(f"채널: **{info['channel']}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("조회수", f"{info['views']:,}")
        m2.metric("좋아요", f"{info['likes']:,}")
        m3.metric("전체 댓글 수", f"{info['comment_count']:,}")
        m4.metric("가져온 댓글 수", f"{len(df):,}")

    st.divider()

    # ── 요약 지표 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("수집 댓글", f"{len(df):,}개")
    c2.metric("총 좋아요", f"{df['like_count'].sum():,}")
    c3.metric("평균 좋아요", f"{df['like_count'].mean():.1f}")
    c4.metric("평균 댓글 길이", f"{df['text_length'].mean():.0f}자")

    # ── 인기 댓글 TOP 15 ──
    st.subheader("👍 좋아요가 많은 댓글 TOP 15")
    top_liked = df.sort_values("like_count", ascending=False).head(15).copy()
    top_liked["label"] = top_liked.apply(
        lambda r: (r["text"][:40] + "…") if len(r["text"]) > 40 else r["text"], axis=1
    )
    fig1 = px.bar(
        top_liked.sort_values("like_count"),
        x="like_count", y="label", orientation="h",
        color="like_count", color_continuous_scale="Reds",
        labels={"like_count": "좋아요 수", "label": ""},
    )
    fig1.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

    # ── 시간대별 댓글 추이 ──
    st.subheader("🕒 시간에 따른 댓글 작성 추이")
    daily = df.set_index("published_at").resample("D").size().reset_index(name="count")
    fig2 = px.area(daily, x="published_at", y="count",
                    labels={"published_at": "날짜", "count": "댓글 수"})
    fig2.update_traces(line_color="#FF0000", fillcolor="rgba(255,0,0,0.2)")
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📏 댓글 길이 분포")
        fig3 = px.histogram(df, x="text_length", nbins=40,
                             labels={"text_length": "댓글 길이(자)"})
        fig3.update_traces(marker_color="#4C78A8")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("🙋 댓글을 많이 남긴 사용자 TOP 10")
        top_authors = df["author"].value_counts().head(10).reset_index()
        top_authors.columns = ["author", "count"]
        fig4 = px.bar(top_authors.sort_values("count"), x="count", y="author", orientation="h",
                       labels={"count": "댓글 수", "author": ""})
        fig4.update_traces(marker_color="#54A24B")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── 워드클라우드 ──
    st.subheader("☁️ 댓글 워드클라우드")

    extra_stopwords = {w.strip() for w in extra_stopwords_input.split(",") if w.strip()}
    stopwords = DEFAULT_STOPWORDS | extra_stopwords

    all_words = []
    for text in df["text"]:
        all_words.extend(tokenize(text, stopwords))

    if not all_words:
        st.info("워드클라우드를 만들 만한 단어가 충분하지 않습니다.")
    else:
        word_freq = Counter(all_words)
        font_path = get_korean_font_path()
        mask = make_circle_mask(900)

        wc = WordCloud(
            width=900, height=900,
            background_color="white",
            mask=mask,
            font_path=font_path if font_path else None,
            colormap=colormap,
            max_words=150,
            prefer_horizontal=0.9,
            relative_scaling=0.5,
            min_font_size=10,
            contour_width=0,
        ).generate_from_frequencies(word_freq)

        wc_array = wc.to_array()
        fig_wc = px.imshow(wc_array)
        fig_wc.update_layout(
            height=650, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_visible=False, yaxis_visible=False,
        )
        st.plotly_chart(fig_wc, use_container_width=True)

        # 상위 단어 막대그래프
        st.subheader("🔤 가장 많이 등장한 단어 TOP 20")
        top_words = pd.DataFrame(word_freq.most_common(20), columns=["word", "count"])
        fig5 = px.bar(top_words.sort_values("count"), x="count", y="word", orientation="h",
                       color="count", color_continuous_scale=colormap,
                       labels={"count": "빈도", "word": ""})
        fig5.update_layout(coloraxis_showscale=False, height=550)
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ── 원본 데이터 ──
    st.subheader("📋 수집된 댓글 데이터")
    st.dataframe(
        df[["author", "text", "like_count", "reply_count", "published_at"]]
        .sort_values("like_count", ascending=False),
        use_container_width=True, height=350,
    )

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("💾 CSV로 다운로드", csv, file_name=f"comments_{video_id}.csv", mime="text/csv")

else:
    st.info("왼쪽 사이드바에 유튜브 URL을 입력하고 **분석 시작** 버튼을 눌러주세요.")
