import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

df = pd.read_excel("/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1127/test_all_evaluate_hybrid.xlsx")


def get_confusion_matrix(df):
    df["gt_intent"] = df["gt_intent"].apply(str)
    df["llm_intent"] = df["llm_intent"].apply(str)

    # 统一收集所有标签
    all_labels = sorted(set(df["gt_intent"]) | set(df["llm_intent"]))

    # 计算混淆矩阵
    cm = confusion_matrix(df["gt_intent"], df["llm_intent"], labels=all_labels)

    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation="vertical")
    plt.title("Confusion Matrix: gt_intent vs llm_intent")
    plt.tight_layout()
    plt.show()

    # 返回 DataFrame 格式的混淆矩阵，并增加每行/每列的 sum、各类别的 precision/recall
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels, dtype="float64")

    # 计算每行/每列的总和
    cm_df["row_sum"] = cm_df.sum(axis=1)
    cm_df.loc["col_sum"] = cm_df.sum(axis=0)

    # 计算每个类别的 precision 和 recall
    # Precision = diag / 列和， Recall = diag / 行和
    diag = cm.diagonal().astype("float64")
    precision = diag / (cm_df.loc["col_sum", all_labels] + 1e-12)
    recall = diag / (cm_df.loc[all_labels, "row_sum"] + 1e-12)

    cm_df.loc["precision", all_labels] = precision
    cm_df.loc[all_labels, "recall"] = recall

    # 保留所有值小数点后4位
    cm_df = cm_df.round(4)
    cm_df.to_excel(
        "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1127/test_all_evaluate_hybrid_confusion_matrix.xlsx"
    )
    print(cm_df)


if __name__ == "__main__":
    get_confusion_matrix(df)
