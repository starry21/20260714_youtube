import re
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageDraw
from wordcloud import WordCloud
from googleapiclient.discovery import build

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="유튜브 댓글 분석기", page_icon="🎬", layout="wide")

YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# ------------------------------------------------------------
# 한글 폰트 준비 (워드클라우드용) - 최초 1회 자동 다운로드 후 캐시
# ------------------------------------------------------------
@st.cache_resource(show_spinner="한글 폰트를 준비하는 중입니다...")
def get_korean_font():
    font_path = "NanumGothic.ttf"
    try:
        import os
        if not os.path.exists(font_path):
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(r.content)
        return font_path
    except Exception:
        return None

FONT_PATH = get_korean_font()

# ------------------------------------------------------------
# 불용어 목록 (필요하면 사이드바에서 추가 가능)
# ------------------------------------------------------------
DEFAULT_STOPWORDS = set("""
그리고 그런데 하지만 그래서 정말 진짜 너무 이거 저거 그거
저는 나는 이런 저런 그런 하는 있는 없는 같은 대한 에서
으로 부터 까지 하고 이고 그냥 근데 이제 아니 네요 어요
있어요 없어요 합니다 있습니다 없습니다 이것 저것 뭔가 정도
사람 우리 당신 여기 거기 저기 오늘 내일 어제 진짜로 완전
그럼 이제는 좀더 좀 더 하나 두개 것도 것은 것을 것이 하면
합니다요 있는데 없는데 같아요 같습니다 영상 채널 구독 댓글
""".split())

POSITIVE_WORDS = set("좋아요 좋다 최고 감사 사랑 훌륭 대박 재밌 재미 웃김 굿 멋있 감동 응원 화이팅 최애".split())
NEGATIVE_WORDS = set("싫어요 싫다 최악 별로 실망 짜증 화남 답답 슬프다 안좋 아쉽 문제".split())

# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
def extract_video_id(url_or_id: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id.strip()):
        return url_or_id.strip()
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_video_info(video_id: str):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    res = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    if not res["items"]:
        return None
    item = res["items"][0]
    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
        "view_count": item["statistics"].get("viewCount", "0"),
        "comment_count": item["statistics"].get("commentCount", "0"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_comments(video_id: str, max_comments: int, order: str):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_token = None

    while len(comments) < max_comments:
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                order=order,
                pageToken=next_token,
                textFormat="plainText",
            )
            res = req.execute()
        except Exception as e:
            raise RuntimeError(str(e))

        for item in res.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": top["authorDisplayName"],
                "text": top["textDisplay"],
                "like_count": top["likeCount"],
                "published_at": top["publishedAt"],
            })

        next_token = res.get("nextPageToken")
        if not next_token:
            break

    return pd.DataFrame(comments)


def tokenize(text: str, min_len: int, extra_stopwords: set):
    words = re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", text)
    stopwords = DEFAULT_STOPWORDS | extra_stopwords
    return [w for w in words if w not in stopwords and len(w) >= min_len]


def simple_sentiment_score(text: str):
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    if pos > neg:
        return "긍정"
    elif neg > pos:
        return "부정"
    return "중립"


def make_circle_mask(size=800):
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=0)
    return np.array(mask)


def color_func_factory():
    palette = ["#4C6EF5", "#22B8CF", "#F59F00", "#845EF7", "#20C997", "#FF6B6B"]
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return palette[random_state.randint(0, len(palette)) if random_state else np.random.randint(0, len(palette))]
    return color_func


# ------------------------------------------------------------
# 사이드바
# ------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
video_input = st.sidebar.text_input("유튜브 영상 URL 또는 ID", placeholder="https://www.youtube.com/watch?v=...")
max_comments = st.sidebar.slider("가져올 댓글 수", 50, 2000, 300, step=50)
order = st.sidebar.selectbox("댓글 정렬", ["relevance", "time"], format_func=lambda x: "인기순" if x == "relevance" else "최신순")
min_word_len = st.sidebar.slider("단어 최소 길이", 1, 4, 2)
extra_stop_input = st.sidebar.text_input("추가 불용어 (쉼표로 구분)", placeholder="예: 구독, 좋아요")
extra_stopwords = set(w.strip() for w in extra_stop_input.split(",") if w.strip())
run_btn = st.sidebar.button("🔍 분석 시작", type="primary", use_container_width=True)

st.title("🎬 유튜브 댓글 분석기")
st.caption("영상 URL을 입력하고 댓글을 분석해서 워드클라우드와 통계를 확인해보세요.")

# ------------------------------------------------------------
# 메인 로직
# ------------------------------------------------------------
if run_btn:
    if not video_input:
        st.error("유튜브 영상 URL 또는 ID를 입력해주세요.")
        st.stop()

    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("올바른 유튜브 URL 또는 ID를 인식하지 못했습니다.")
        st.stop()

    with st.spinner("영상 정보를 불러오는 중입니다..."):
        info = get_video_info(video_id)

    if info is None:
        st.error("영상 정보를 찾을 수 없습니다. URL을 다시 확인해주세요.")
        st.stop()

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(info["thumbnail"], use_container_width=True)
    with col2:
        st.subheader(info["title"])
        st.write(f"📺 채널: **{info['channel']}**")
        m1, m2 = st.columns(2)
        m1.metric("조회수", f"{int(info['view_count']):,}")
        m2.metric("전체 댓글수", f"{int(info['comment_count']):,}")

    st.divider()

    try:
        with st.spinner("댓글을 가져오는 중입니다... (댓글 수가 많으면 시간이 걸릴 수 있어요)"):
            df = fetch_comments(video_id, max_comments, order)
    except RuntimeError as e:
        if "commentsDisabled" in str(e):
            st.error("이 영상은 댓글 기능이 비활성화되어 있습니다.")
        else:
            st.error(f"댓글을 가져오는 중 오류가 발생했습니다: {e}")
        st.stop()

    if df.empty:
        st.warning("가져온 댓글이 없습니다.")
        st.stop()

    df["length"] = df["text"].str.len()
    df["sentiment"] = df["text"].apply(simple_sentiment_score)

    # ---------------- 통계 카드 ----------------
    st.subheader("📊 댓글 요약 통계")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("수집한 댓글 수", f"{len(df):,}")
    c2.metric("평균 좋아요", f"{df['like_count'].mean():.1f}")
    c3.metric("평균 글자수", f"{df['length'].mean():.1f}자")
    c4.metric("최다 좋아요", f"{df['like_count'].max():,}")

    st.divider()

    # ---------------- 워드클라우드 ----------------
    st.subheader("☁️ 댓글 워드클라우드")

    all_words = []
    for t in df["text"]:
        all_words.extend(tokenize(t, min_word_len, extra_stopwords))

    if not all_words:
        st.warning("워드클라우드를 만들 단어가 충분하지 않습니다. 불용어 설정을 확인해보세요.")
    else:
        word_freq = Counter(all_words)

        wc_kwargs = dict(
            width=1000,
            height=1000,
            background_color="white",
            mask=make_circle_mask(1000),
            max_words=150,
            prefer_horizontal=0.9,
            relative_scaling=0.4,
            color_func=color_func_factory(),
        )
        if FONT_PATH:
            wc_kwargs["font_path"] = FONT_PATH
        else:
            st.info("한글 폰트를 자동으로 불러오지 못해 영어 위주로 표시될 수 있습니다.")

        wc = WordCloud(**wc_kwargs).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        st.pyplot(fig, use_container_width=True)

        # ---------------- 상위 키워드 바 차트 ----------------
        st.subheader("🔑 상위 키워드 TOP 20")
        top_words = word_freq.most_common(20)
        top_df = pd.DataFrame(top_words, columns=["단어", "빈도"])
        st.bar_chart(top_df.set_index("단어"))

    st.divider()

    # ---------------- 감정 분포 ----------------
    st.subheader("💬 간단 감정 분포 (참고용 키워드 기반)")
    st.caption("정교한 감정분석 모델이 아니라 키워드 매칭 기반의 참고 지표입니다.")
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.divider()

    # ---------------- 인기 댓글 ----------------
    st.subheader("🏆 좋아요 많은 댓글 TOP 10")
    top_comments = df.sort_values("like_count", ascending=False).head(10)
    for _, row in top_comments.iterrows():
        st.markdown(f"**{row['author']}** · 👍 {row['like_count']}")
        st.write(row["text"])
        st.markdown("---")

    # ---------------- 원본 데이터 & 다운로드 ----------------
    st.subheader("📋 전체 댓글 데이터")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 CSV로 다운로드",
        data=csv,
        file_name=f"comments_{video_id}.csv",
        mime="text/csv",
    )
else:
    st.info("왼쪽 사이드바에 유튜브 URL을 입력하고 '분석 시작'을 눌러주세요.")

