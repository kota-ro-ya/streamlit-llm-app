# 環境変数の読み込み(これでAPIキーが読み込める)
from dotenv import load_dotenv
load_dotenv()

# OpenAIのAPIキーを読み込み
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChainとOpenAIのインポート
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import streamlit as st

# ===== 定数定義 =====
class ExpertTypes:
    """専門家タイプの定数"""
    FITNESS = "筋トレ専門家"
    DIET = "ダイエット専門家"
    
    @classmethod
    def get_all(cls):
        return [cls.FITNESS, cls.DIET]

class LLMConfig:
    """LLM設定の定数"""
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.7
    TIMEOUT = 60

class ExpertPrompts:
    """専門家プロンプトの定数"""
    FITNESS_PROMPT = """あなたは経験豊富な筋トレ専門家です。以下の点を考慮して回答してください：

1. ユーザーのレベル（初心者・中級者・上級者）に応じた適切なアドバイス
2. 安全で効果的なトレーニング方法の提案
3. 具体的な回数・セット数・重量の目安
4. フォームの重要性と注意点
5. 栄養・休息の重要性
6. 怪我の予防方法

専門的で実用的なアドバイスを、親しみやすく分かりやすい言葉で提供してください。"""

    DIET_PROMPT = """あなたは経験豊富なダイエット専門家・栄養士です。以下の点を考慮して回答してください：

1. 健康的で持続可能なダイエット方法の提案
2. 個人の体質・ライフスタイルに合わせたアドバイス
3. 栄養バランスの取れた食事プラン
4. 適切な運動量とタイミング
5. ストレス管理と睡眠の重要性
6. 無理のない目標設定と進捗管理

医学的根拠に基づいた安全で効果的なアドバイスを、親しみやすく分かりやすい言葉で提供してください。"""

class InputLabels:
    """入力ラベルの定数"""
    FITNESS_LABEL = "筋トレについて相談したい内容を入力してください。\n例：\n- 初心者におすすめの筋トレメニューは？\n- 腕を太くしたい\n- 自宅でできる筋トレを教えて"
    DIET_LABEL = "ダイエットについて相談したい内容を入力してください。\n例：\n- 健康的に痩せる方法は？\n- 食事制限のコツを教えて\n- 運動と食事のバランスは？"

# ページ設定
st.set_page_config(
    page_title="筋トレ・ダイエット相談アプリ",
    page_icon="💪",
    layout="wide"
)

st.title("💪 筋トレ・ダイエット相談アプリ")
st.write("専門家に筋トレやダイエットについて相談できます。入力フォームにテキストを入力し、「実行ボタン」を押してください。")

# 専門家の選択
selected_item = st.radio("どちらの専門家に相談したいですか？", ExpertTypes.get_all())

st.divider()

# 動的な入力フォーム生成
def get_input_label(expert_type: str) -> str:
    """専門家タイプに応じた入力ラベルを取得"""
    if expert_type == ExpertTypes.FITNESS:
        return InputLabels.FITNESS_LABEL
    elif expert_type == ExpertTypes.DIET:
        return InputLabels.DIET_LABEL
    else:
        return f"{expert_type}について相談したい内容を入力してください。"

input_message = st.text_area(
    get_input_label(selected_item),
    height=100
)

# 専門家のプロンプト定義
def get_expert_prompt(expert_type: str) -> str:
    """専門家タイプに応じたプロンプトを取得"""
    if expert_type == ExpertTypes.FITNESS:
        return ExpertPrompts.FITNESS_PROMPT
    elif expert_type == ExpertTypes.DIET:
        return ExpertPrompts.DIET_PROMPT
    else:
        return f"あなたは{expert_type}です。専門的なアドバイスを提供してください。"

# LLM設定の一元管理
def create_llm_client() -> ChatOpenAI:
    """LLMクライアントを作成"""
    return ChatOpenAI(
        model=LLMConfig.MODEL,
        temperature=LLMConfig.TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        request_timeout=LLMConfig.TIMEOUT
    )

# 入力値検証
def validate_input(user_input: str) -> tuple[bool, str]:
    """入力値の検証"""
    if not user_input or not user_input.strip():
        return False, "相談内容を入力してください。"
    
    if len(user_input.strip()) < 5:
        return False, "もう少し詳しく相談内容を入力してください。"
    
    if len(user_input) > 1000:
        return False, "相談内容が長すぎます。1000文字以内で入力してください。"
    
    return True, ""

# ✅ LLMに問い合わせる関数（条件③）
def get_expert_answer(expert_type: str, user_input: str, llm: ChatOpenAI) -> str:
    """専門家から回答を取得"""
    system_prompt = get_expert_prompt(expert_type)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    response = llm.invoke(messages)
    return response.content

# 実行ボタン
if st.button("🤖 専門家に相談する", type="primary"):
    # 入力値検証
    is_valid, error_message = validate_input(input_message)
    
    if not is_valid:
        st.warning(f"⚠️ {error_message}")
    elif not OPENAI_API_KEY:
        st.error("❌ 設定が完了していません。.envファイルにAPIキーを設定してください。")
    else:
        with st.spinner("専門家が回答を準備中..."):
            try:
                # LLMクライアントを作成
                llm = create_llm_client()
                
                # 専門家から回答を取得
                answer = get_expert_answer(selected_item, input_message, llm)

                # 結果表示
                st.divider()
                st.subheader(f"🎯 {selected_item}の回答")
                st.write(answer)

                st.info("💡 この回答は一般的なアドバイスです。個別の健康状態や目標については、専門医やトレーナーに直接相談することをお勧めします。")

            except Exception as e:
                st.error("❌ エラーが発生しました。設定を確認してください。")
                st.error(f"デバッグ情報: {str(e)}")

# サイドバーに追加情報
with st.sidebar:
    st.header("📋 使い方")
    st.markdown("""
    1. 専門家を選択
    2. 相談内容を入力
    3. 「専門家に相談する」ボタンをクリック
    4. AIが専門的なアドバイスを提供
    """)
    
    st.header("🔧 設定")
    if OPENAI_API_KEY:
        st.success("✅ 設定完了")
    else:
        st.warning("⚠️ 設定が必要です")
        st.write(".envファイルにAPIキーを設定してください")
    
    st.header("⚠️ 注意事項")
    st.markdown("""
    - このアプリは一般的なアドバイスを提供します
    - 個別の健康状態については専門医に相談してください
    - 怪我のリスクがある場合は無理をしないでください
    """)

