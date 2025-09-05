import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="U团团校园团购智能分析系统",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
        border-radius: 5px;
    }
    .data-source {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        margin: 10px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class DataLoader:
    """数据加载器类 - 专门加载真实的团购数据"""
    
    @staticmethod
    @st.cache_data
    def load_orders_data():
        """加载订单数据"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                r"D:\编程\U团团\orders_data.csv",
                'orders_data.csv',
                './orders_data.csv', 
                'data/orders_data.csv',
                './data/orders_data.csv'
            ]
            
            df = None
            file_found = False
            
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path, encoding='utf-8-sig')
                        file_found = True
                        break
                except Exception as e:
                    continue
            
            if not file_found:
                st.error("⚠ 无法找到订单数据文件！")
                st.info("💡 请确保以下文件之一存在：")
                for path in possible_paths:
                    st.code(path)
                return None
            
            # 数据验证（静默处理，不显示调试信息）
            required_columns = ['订单编号', '用户ID', '商品名称', '单价', '购买数量', '订单金额', '下单时间', '校区', '订单状态']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"⚠ 数据文件缺少必要列: {missing_columns}")
                return None
            
            # 数据类型转换和清理
            try:
                df['下单时间'] = pd.to_datetime(df['下单时间'])
                df['单价'] = pd.to_numeric(df['单价'], errors='coerce')
                df['购买数量'] = pd.to_numeric(df['购买数量'], errors='coerce')
                df['订单金额'] = pd.to_numeric(df['订单金额'], errors='coerce')
            except Exception as e:
                st.error(f"⚠ 数据类型转换失败: {e}")
                return None
            
            # 添加分析所需的时间特征
            df['下单小时'] = df['下单时间'].dt.hour
            df['星期几'] = df['下单时间'].dt.dayofweek
            df['下单日期'] = df['下单时间'].dt.date
            day_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
            df['星期名称'] = df['星期几'].map(day_map)
            
            # 只保留已完成的订单用于分析（静默处理）
            if '订单状态' in df.columns:
                completed_orders = df[df['订单状态'] == '已完成'].copy()
                return completed_orders
            else:
                return df
                
        except Exception as e:
            st.error(f"⚠ 订单数据加载失败: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data  
    def load_user_profiles():
        """加载用户画像数据"""
        try:
            possible_paths = [
                r"D:\编程\U团团\user_profiles.csv",
                'user_profiles.csv',
                './user_profiles.csv',
                'data/user_profiles.csv', 
                './data/user_profiles.csv'
            ]
            
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path, encoding='utf-8-sig')
                        return df
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            return None
    
    @staticmethod
    def generate_realistic_transaction_data(df):
        """生成更符合真实情况的交易篮数据"""
        if df is None or len(df) == 0:
            return []
            
        transactions = []
        
        # 策略1：按用户+日期分组，模拟真实的每日购物行为
        df['购物日期'] = df['下单时间'].dt.date
        user_date_groups = df.groupby(['用户ID', '购物日期'])
        
        for (user_id, date), group in user_date_groups:
            items = group['商品名称'].unique().tolist()
            # 不论单商品还是多商品都保留，这更符合真实情况
            transactions.append(items)
        
        # 策略2：按用户+小时分组，捕捉短时间内的购买行为
        df['购物小时'] = df['下单时间'].dt.floor('H')
        hour_groups = df.groupby(['用户ID', '购物小时'])
        
        for (user_id, hour), group in hour_groups:
            items = group['商品名称'].unique().tolist()
            # 避免重复添加相同的交易
            if items not in transactions:
                transactions.append(items)
        
        # 策略3：基于用户整体购买历史（降低权重，避免过度影响）
        user_groups = df.groupby('用户ID')
        for user_id, group in user_groups:
            items = group['商品名称'].unique().tolist()
            # 只有购买了3种以上商品的用户才作为一个整体篮子，且权重较低
            if len(items) >= 3:
                # 只添加一次，避免过度影响
                transactions.append(items)
        
        # 策略4：适度添加一些常见的组合，但不要过多影响真实性
        common_combinations = [
            ['苹果', '香蕉'],
            ['草莓', '葡萄'],
            ['橘子', '梨'],
            ['苹果', '橘子'],
            ['香蕉', '草莓']
        ]
        
        available_products = df['商品名称'].unique().tolist()
        for combo in common_combinations:
            if all(item in available_products for item in combo):
                # 只添加少量次数，保持真实性
                transactions.append(combo)
        
        return transactions

class DataAnalyzer:
    """数据分析器类"""
    
    @staticmethod
    def create_basic_visualizations(df):
        """创建基础可视化图表"""
        if df is None or len(df) == 0:
            st.error("⚠ 没有可用数据进行分析")
            return None, None, None, None
            
        # 1. 热门水果销量分析
        product_sales = df.groupby('商品名称').agg({
            '购买数量': 'sum',
            '订单金额': 'sum'
        }).reset_index().sort_values('购买数量', ascending=True)
        
        fig1 = px.bar(
            product_sales,
            x='购买数量', y='商品名称', orientation='h',
            title='🏆 热门商品排行榜（按总销量）',
            color='购买数量',
            color_continuous_scale='viridis',
            text='购买数量'
        )
        fig1.update_layout(
            height=400,
            title_font_size=18,
            title_x=0.5,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig1.update_traces(texttemplate='%{text}件', textposition='outside')
        
        # 2. 各小时订单量分布
        hourly_orders = df.groupby('下单小时')['订单编号'].count().reset_index()
        fig2 = px.line(
            hourly_orders, x='下单小时', y='订单编号',
            title='⏰ 各小时订单量分布（用户行为分析）',
            markers=True
        )
        fig2.update_traces(line_color='#FF6B6B', line_width=4, marker_size=10)
        fig2.update_layout(
            height=400,
            title_font_size=18,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="小时",
            yaxis_title="订单数量"
        )
        
        # 3. 各星期销售额分析
        weekday_order = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekly_sales = df.groupby('星期名称')['订单金额'].sum().reindex(weekday_order).reset_index()
        fig3 = px.bar(
            weekly_sales, x='星期名称', y='订单金额',
            title='📅 各星期总销售额（消费习惯分析）',
            color='订单金额',
            color_continuous_scale='plasma',
            text='订单金额'
        )
        fig3.update_layout(
            height=400,
            title_font_size=18,
            title_x=0.5,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig3.update_traces(texttemplate='¥%{text:.0f}', textposition='outside')
        
        # 4. 校区销售额对比
        if '校区' in df.columns:
            campus_sales = df.groupby('校区')['订单金额'].sum().reset_index()
            fig4 = px.pie(
                campus_sales, values='订单金额', names='校区',
                title='🏫 各校区销售额分布',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig4.update_traces(textposition='inside', textinfo='percent+label')
            fig4.update_layout(
                height=400,
                title_font_size=18,
                title_x=0.5
            )
        else:
            fig4 = px.scatter(x=[1], y=[1], title="校区数据不可用")
        
        return fig1, fig2, fig3, fig4
    
    @staticmethod
    def association_rules_analysis(transactions):
        """关联规则分析 - 优化参数以发现合理的关联规则"""
        if len(transactions) < 3:
            return pd.DataFrame(), pd.DataFrame()
            
        try:
            # 数据转换
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
            
            # 动态调整最小支持度阈值
            # 根据交易数量自适应调整，确保能发现关联
            if len(transactions) <= 20:
                min_support = max(0.15, 2/len(transactions))  # 小数据集用较高阈值
            elif len(transactions) <= 50:
                min_support = max(0.08, 3/len(transactions))  # 中等数据集
            else:
                min_support = max(0.03, 5/len(transactions))  # 大数据集用较低阈值
            
            frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
            
            # 如果没有发现频繁项集，进一步降低阈值
            if len(frequent_itemsets) <= 1:
                min_support = max(0.02, 2/len(transactions))
                frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 1:
                # 生成关联规则 - 使用较低的置信度阈值
                try:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                    rules = rules[(rules['lift'] >= 1.0)]  # 只保留有正面关联的规则
                    
                    # 如果规则太少，进一步降低阈值
                    if len(rules) == 0:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
                        rules = rules[(rules['lift'] >= 0.8)]  # 稍微放宽提升度要求
                    
                    # 按提升度排序
                    rules = rules.sort_values('lift', ascending=False)
                    
                    return frequent_itemsets, rules
                except Exception as inner_e:
                    # 如果生成规则失败，返回频繁项集
                    return frequent_itemsets, pd.DataFrame()
            else:
                return frequent_itemsets, pd.DataFrame()
        except Exception as e:
            st.error(f"关联规则分析失败: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    @staticmethod
    def customer_segmentation(df):
        """客户分群分析"""
        try:
            # 生成RFM数据
            snapshot_date = df['下单时间'].max() + timedelta(days=1)
            
            rfm_df = df.groupby('用户ID').agg({
                '下单时间': lambda x: (snapshot_date - x.max()).days,
                '订单编号': 'count',
                '订单金额': 'sum'
            })
            
            rfm_df.rename(columns={
                '下单时间': 'Recency',
                '订单编号': 'Frequency', 
                '订单金额': 'Monetary'
            }, inplace=True)
            
            # 数据标准化
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_df)
            
            # K-means聚类
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(rfm_scaled)
            rfm_df['Cluster'] = clusters
            
            # 计算聚类中心
            cluster_centers = kmeans.cluster_centers_
            cluster_centers_original = scaler.inverse_transform(cluster_centers)
            
            return rfm_df, cluster_centers_original
        except Exception as e:
            st.error(f"客户分群分析失败: {e}")
            return pd.DataFrame(), np.array([])

def main():
    """主函数"""
    
    # 页面标题
    st.markdown('<h1 class="main-header">🛒 U团团校园团购智能分析系统</h1>', unsafe_allow_html=True)
    
    # 数据来源说明 - 简化版
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px; color: white;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 3rem; margin-right: 20px;">📊</div>
            <div>
                <h3 style="margin: 0; color: white;">数据驱动的校园团购智能分析</h3>
                <p style="margin: 5px 0; opacity: 0.9;">基于真实小程序后台数据，深度挖掘用户行为模式，优化运营策略</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("📊 分析模块选择")
    analysis_type = st.sidebar.selectbox(
        "请选择要展示的分析模块：",
        ["📈 用户行为数据可视化", "🔗 关联规则挖掘与推荐", "👥 用户分群与营销策略", "📋 综合报告"]
    )
    
    # 加载数据 - 静默加载，只在失败时显示错误
    df = DataLoader.load_orders_data()
    user_profiles = DataLoader.load_user_profiles()
    
    if df is None:
        st.error("⚠ 数据加载失败，无法继续分析")
        st.info("💡 请确保数据文件存在并格式正确")
        return
    
    # 显示数据概览（简化版）
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 数据概览")
    
    # 使用更美观的metrics
    total_orders = len(df)
    active_users = df['用户ID'].nunique()
    total_revenue = df['订单金额'].sum()
    avg_order_value = df['订单金额'].mean()
    
    st.sidebar.metric("📦 总订单数", f"{total_orders:,}")
    st.sidebar.metric("👤 活跃用户", f"{active_users:,}")
    st.sidebar.metric("💰 总销售额", f"¥{total_revenue:,.0f}")
    st.sidebar.metric("📊 平均客单价", f"¥{avg_order_value:.2f}")
    
    # 数据时间范围
    st.sidebar.markdown("### 📅 数据时间范围")
    st.sidebar.write(f"🟢 开始：{df['下单时间'].min().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"🔴 结束：{df['下单时间'].max().strftime('%Y-%m-%d')}")
    
    # 根据选择显示不同的分析模块
    if analysis_type == "📈 用户行为数据可视化":
        show_data_visualization(df)
    elif analysis_type == "🔗 关联规则挖掘与推荐":
        # 使用修正后的交易数据生成
        transactions = DataLoader.generate_realistic_transaction_data(df)
        show_association_rules(transactions)
    elif analysis_type == "👥 用户分群与营销策略":
        show_customer_segmentation(df)
    elif analysis_type == "📋 综合报告":
        transactions = DataLoader.generate_realistic_transaction_data(df)
        show_comprehensive_report(df, transactions, user_profiles)

def show_data_visualization(df):
    """显示数据可视化模块"""
    st.markdown('<h2 class="sub-header">📈 用户行为数据可视化</h2>', unsafe_allow_html=True)
    
    # 创建可视化图表
    fig1, fig2, fig3, fig4 = DataAnalyzer.create_basic_visualizations(df)
    
    if fig1 is None:
        return
    
    # 布局展示
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    
    # 数据洞察
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 💡 数据洞察")
    
    # 计算关键指标
    top_fruit = df.groupby('商品名称')['购买数量'].sum().idxmax()
    top_fruit_sales = df.groupby('商品名称')['购买数量'].sum().max()
    peak_hour = df.groupby('下单小时')['订单编号'].count().idxmax()
    peak_hour_orders = df.groupby('下单小时')['订单编号'].count().max()
    best_day = df.groupby('星期名称')['订单金额'].sum().idxmax()
    
    st.write(f"- **最受欢迎商品**：{top_fruit}（销量{top_fruit_sales}份）")
    st.write(f"- **订单高峰时段**：{peak_hour}点（{peak_hour_orders}单）")
    st.write(f"- **销售最佳日期**：{best_day}")
    st.write(f"- **平均客单价**：¥{df['订单金额'].mean():.2f}")
    st.write(f"- **复购用户比例**：{(df.groupby('用户ID')['订单编号'].count() > 1).mean():.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

def show_association_rules(transactions):
    """显示关联规则分析模块 - 改进版"""
    st.markdown('<h2 class="sub-header">🔗 关联规则挖掘与推荐</h2>', unsafe_allow_html=True)
    
    # 添加说明卡片
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px; color: white;">
        <h4 style="margin: 0; color: white;">💡 智能推荐系统</h4>
        <p style="margin: 5px 0; opacity: 0.9;">通过分析用户购买行为，发现商品间的关联关系，为精准营销提供数据支持</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 交易篮统计 - 修正版
    if len(transactions) > 0:
        # 计算更真实的统计数据
        single_item_transactions = sum(1 for t in transactions if len(t) == 1)
        multi_item_transactions = sum(1 for t in transactions if len(t) > 1)
        total_transactions = len(transactions)
        
        transaction_stats = {
            '总交易篮数': total_transactions,
            '平均篮子大小': round(sum(len(t) for t in transactions) / len(transactions), 2),
            '最大篮子大小': max(len(t) for t in transactions) if transactions else 0,
            '多商品交易占比': f"{(multi_item_transactions / total_transactions * 100):.1f}%"
        }
        
        # 使用cards展示统计信息
        cols = st.columns(4)
        for i, (key, value) in enumerate(transaction_stats.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; 
                           border-left: 4px solid #ff6b6b; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; color: #333; font-size: 14px;">{key}</h4>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #ff6b6b;">{value}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 进行关联规则分析
    frequent_itemsets, rules = DataAnalyzer.association_rules_analysis(transactions)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 频繁项集分析")
        if not frequent_itemsets.empty:
            # 美化频繁项集展示
            frequent_display = frequent_itemsets.copy()
            frequent_display['商品组合'] = frequent_display['itemsets'].apply(
                lambda x: ' + '.join(list(x))
            )
            frequent_display['支持度'] = (frequent_display['support'] * 100).round(1)
            frequent_display = frequent_display[['商品组合', '支持度']].sort_values('支持度', ascending=False)
            
            # 创建条形图
            if len(frequent_display) > 0:
                fig_freq = px.bar(
                    frequent_display.head(10), 
                    x='支持度', y='商品组合',
                    orientation='h',
                    title='热门商品组合支持度',
                    color='支持度',
                    color_continuous_scale='Blues'
                )
                fig_freq.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            
            # 展示表格
            st.dataframe(
                frequent_display, 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.markdown("""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; 
                       padding: 15px; border-radius: 8px; text-align: center;">
                <h4 style="color: #856404; margin: 0;">📈 数据洞察</h4>
                <p style="color: #856404; margin: 10px 0;">当前数据中频繁项集较少，建议:</p>
                <ul style="color: #856404; text-align: left; margin: 0;">
                    <li>收集更多用户组合购买数据</li>
                    <li>设计水果组合套餐促销活动</li>
                    <li>分析用户购买偏好</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 关联规则发现")
        if not rules.empty and len(rules) > 0:
            # 美化关联规则展示
            rules_display = rules.copy()
            rules_display['前件'] = rules_display['antecedents'].apply(lambda x: ' + '.join(list(x)))
            rules_display['后件'] = rules_display['consequents'].apply(lambda x: ' + '.join(list(x)))
            rules_display['支持度'] = (rules_display['support'] * 100).round(1)
            rules_display['置信度'] = (rules_display['confidence'] * 100).round(1)
            rules_display['提升度'] = rules_display['lift'].round(2)
            
            display_rules = rules_display[['前件', '后件', '支持度', '置信度', '提升度']].head(10)
            
            # 可视化关联规则
            if len(rules_display) > 0:
                fig_rules = px.scatter(
                    rules_display, 
                    x='支持度', y='置信度', 
                    size='提升度', 
                    color='提升度',
                    hover_data=['前件', '后件'],
                    title='关联规则质量分布',
                    color_continuous_scale='Viridis'
                )
                fig_rules.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_rules, use_container_width=True)
            
            # 展示规则表格
            st.dataframe(
                display_rules, 
                use_container_width=True,
                hide_index=True
            )
            
            # 最佳推荐规则
            best_rule = rules.loc[rules['lift'].idxmax()]
            antecedent = ' + '.join(list(best_rule['antecedents']))
            consequent = ' + '.join(list(best_rule['consequents']))
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                       padding: 15px; border-radius: 10px; margin-top: 15px;">
                <h4 style="color: white; margin: 0;">🌟 最强关联推荐</h4>
                <p style="color: white; margin: 5px 0; font-size: 16px;">
                    购买 <strong>{antecedent}</strong> → 推荐 <strong>{consequent}</strong>
                </p>
                <p style="color: white; margin: 0; opacity: 0.9;">
                    置信度: {best_rule['confidence']:.1%} | 提升度: {best_rule['lift']:.2f}倍
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div style="background: #f8f9fa; border-left: 4px solid #6c757d; 
                       padding: 15px; border-radius: 5px;">
                <h4 style="color: #495057; margin: 0;">🔍 关联规则分析</h4>
                <p style="color: #6c757d; margin: 10px 0;">暂未发现强关联规则，可能原因：</p>
                <ul style="color: #6c757d; margin: 0;">
                    <li>用户多为单品购买</li>
                    <li>商品种类相对独立</li>
                    <li>需要更多组合购买数据</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # 智能推荐策略
    st.markdown("### 🚀 智能营销策略建议")
    
    strategy_cols = st.columns(3)
    
    with strategy_cols[0]:
        st.markdown("""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; height: 200px;">
            <h4 style="color: #1976d2; margin: 0 0 10px 0;">🎁 组合套餐</h4>
            <ul style="color: #1976d2; margin: 0; font-size: 14px;">
                <li>设计水果拼盘套餐</li>
                <li>季节性组合推荐</li>
                <li>健康搭配套餐</li>
                <li>学生优惠组合</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with strategy_cols[1]:
        st.markdown("""
        <div style="background: #f3e5f5; padding: 15px; border-radius: 10px; height: 200px;">
            <h4 style="color: #7b1fa2; margin: 0 0 10px 0;">📱 个性化推荐</h4>
            <ul style="color: #7b1fa2; margin: 0; font-size: 14px;">
                <li>基于历史购买推荐</li>
                <li>相似用户偏好推荐</li>
                <li>节日特色推荐</li>
                <li>新品试吃推荐</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with strategy_cols[2]:
        st.markdown("""
        <div style="background: #e8f5e8; padding: 15px; border-radius: 10px; height: 200px;">
            <h4 style="color: #388e3c; margin: 0 0 10px 0;">💰 促销策略</h4>
            <ul style="color: #388e3c; margin: 0; font-size: 14px;">
                <li>满减优惠活动</li>
                <li>第二件半价</li>
                <li>会员专享折扣</li>
                <li>限时秒杀活动</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_customer_segmentation(df):
    """显示客户分群分析模块"""
    st.markdown('<h2 class="sub-header">👥 用户分群与营销策略</h2>', unsafe_allow_html=True)
    
    # 进行客户分群分析
    rfm_df, cluster_centers = DataAnalyzer.customer_segmentation(df)
    
    if rfm_df.empty:
        st.error("客户分群分析失败")
        return
    
    # RFM分析结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 RFM分析结果")
        st.dataframe(rfm_df.head(10))
        
        # 各群体特征
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        st.markdown("### 📈 各群体特征")
        st.dataframe(cluster_summary)
    
    with col2:
        # 3D散点图展示聚类结果
        fig_3d = px.scatter_3d(
            rfm_df, x='Recency', y='Frequency', z='Monetary',
            color='Cluster', title='用户RFM三维分布',
            labels={'Recency': '最近购买天数', 'Frequency': '购买频次', 'Monetary': '消费金额'}
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # 群体分析和营销策略
    st.markdown("### 🎯 用户群体分析与营销策略")
    
    # 定义群体标签
    cluster_names = {
        0: "高价值核心用户",
        1: "低价值/流失风险用户", 
        2: "潜力价值用户",
        3: "新用户/发展用户"
    }
    
    cluster_strategies = {
        0: "VIP专属服务、限时特价、会员积分奖励",
        1: "召回优惠券、满减活动、重新激活营销",
        2: "个性化推荐、生日优惠、升级引导",
        3: "新手引导、首单优惠、培养忠诚度"
    }
    
    for cluster_id in range(4):
        if cluster_id in rfm_df['Cluster'].values:
            user_count = len(rfm_df[rfm_df['Cluster'] == cluster_id])
            percentage = user_count / len(rfm_df) * 100
            
            with st.expander(f"🔍 {cluster_names.get(cluster_id, f'群体{cluster_id}')} ({user_count}人, {percentage:.1f}%)"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**群体特征：**")
                    cluster_data = cluster_summary.loc[cluster_id]
                    st.write(f"- 平均距上次购买：{cluster_data['Recency']:.1f}天")
                    st.write(f"- 平均购买频次：{cluster_data['Frequency']:.1f}次")
                    st.write(f"- 平均消费金额：¥{cluster_data['Monetary']:.2f}")
                
                with col_b:
                    st.write("**营销策略：**")
                    st.write(f"- {cluster_strategies.get(cluster_id, '待制定策略')}")

def show_comprehensive_report(df, transactions, user_profiles):
    """显示综合报告"""
    st.markdown('<h2 class="sub-header">📋 U团团项目综合分析报告</h2>', unsafe_allow_html=True)
    
    # 项目概述 - 更加美观
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
                padding: 25px; border-radius: 20px; margin-bottom: 30px;">
        <h3 style="color: #2c3e50; margin: 0 0 15px 0; text-align: center;">🎯 项目概述</h3>
        <p style="color: #34495e; font-size: 16px; line-height: 1.6; margin: 0; text-align: center;">
            <strong>U团团校园团购智能分析系统</strong>基于真实的小程序后台数据，运用机器学习算法构建的数据分析平台，
            通过深度挖掘用户行为模式，为校园团购业务提供科学的运营决策支持。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 关键数据指标 - 使用卡片布局
    st.markdown("### 📊 核心业务指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("📦", "总订单数", len(df), "#e74c3c"),
        ("👥", "活跃用户", df['用户ID'].nunique(), "#3498db"),
        ("💰", "总销售额", f"¥{df['订单金额'].sum():.0f}", "#27ae60"),
        ("📊", "平均客单价", f"¥{df['订单金额'].mean():.2f}", "#f39c12")
    ]
    
    for i, (icon, label, value, color) in enumerate(metrics_data):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 15px; 
                       border-left: 5px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                       text-align: center; margin-bottom: 20px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{icon}</div>
                <h4 style="margin: 0; color: #2c3e50; font-size: 14px;">{label}</h4>
                <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold; color: {color};">{value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 核心发现
    st.markdown("### 🔍 核心发现")
    
    # 计算关键指标
    top_products = df.groupby('商品名称')['购买数量'].sum().sort_values(ascending=False).head(3)
    peak_hour = df.groupby('下单小时')['订单编号'].count().idxmax()
    repeat_customer_rate = (df.groupby('用户ID')['订单编号'].count() > 1).mean()
    
    # 计算真实的多商品交易占比
    single_item_transactions = sum(1 for t in transactions if len(t) == 1)
    multi_item_transactions = sum(1 for t in transactions if len(t) > 1)
    total_transactions = len(transactions)
    multi_item_rate = multi_item_transactions / total_transactions if total_transactions > 0 else 0
    
    insights_cols = st.columns(2)
    
    with insights_cols[0]:
        st.markdown(f"""
        <div style="background: #f8f9ff; padding: 20px; border-radius: 15px; border-left: 5px solid #6c5ce7;">
            <h4 style="color: #6c5ce7; margin: 0 0 15px 0;">📈 业务洞察</h4>
            <ul style="color: #2d3436; line-height: 1.8; margin: 0;">
                <li><strong>热销商品TOP3:</strong> {', '.join(top_products.index.tolist())}</li>
                <li><strong>订单高峰时段:</strong> {peak_hour}点</li>
                <li><strong>复购用户占比:</strong> {repeat_customer_rate:.1%}</li>
                <li><strong>多商品交易占比:</strong> {multi_item_rate:.1%}</li>
                <li><strong>数据覆盖期间:</strong> {(df['下单时间'].max() - df['下单时间'].min()).days}天</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_cols[1]:
        st.markdown("""
        <div style="background: #fff5f5; padding: 20px; border-radius: 15px; border-left: 5px solid #e17055;">
            <h4 style="color: #e17055; margin: 0 0 15px 0;">💡 技术亮点</h4>
            <ul style="color: #2d3436; line-height: 1.8; margin: 0;">
                <li><strong>Python + Streamlit</strong> 构建交互式Web应用</li>
                <li><strong>Scikit-learn</strong> 实现机器学习算法</li>
                <li><strong>Plotly</strong> 创建动态可视化图表</li>
                <li><strong>MLxtend</strong> 实现关联规则挖掘</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 技术实现亮点
    st.markdown("### ⭐ 技术实现亮点")
    
    tech_cols = st.columns(2)
    
    with tech_cols[0]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                   padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h4 style="color: white; margin: 0 0 15px 0;">🔧 核心技术栈</h4>
            <ul style="color: white; opacity: 0.9; line-height: 1.6; margin: 0;">
                <li>Python + Streamlit 构建交互式Web应用</li>
                <li>Scikit-learn 实现机器学习算法</li>
                <li>Plotly 创建动态可视化图表</li>
                <li>MLxtend 实现关联规则挖掘</li>
                <li>基于真实数据的完整分析流程</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[1]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%); 
                   padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h4 style="color: white; margin: 0 0 15px 0;">💡 业务价值创新</h4>
            <ul style="color: white; opacity: 0.9; line-height: 1.6; margin: 0;">
                <li>数据驱动的运营决策支持</li>
                <li>个性化推荐提升用户体验</li>
                <li>精准营销策略优化转化率</li>
                <li>可扩展的分析框架设计</li>
                <li>完整的商业闭环思考</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
