import json
import os
import xml.etree.ElementTree as ET
from pprint import pprint

import langchain
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatTongyi
from lxml.etree import SubElement, Element
from lxml.html import tostring

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatTongyi(model_kwargs={"temperature": 0.7})


def xml_to_dataframe(xml_data):
    # 解析XML数据
    root = ET.fromstring(xml_data)

    # 递归函数来处理XML元素
    def parse_element(element):
        # 初始化一个字典来存储元素数据
        data = {}

        # 遍历元素的子元素
        for child in element:
            # 递归处理子元素
            child_data = parse_element(child)

            # 如果子元素有多个，需要将它们合并到列表中
            if child.tag in data:
                if type(data[child.tag]) is list:
                    data[child.tag].append(child_data)
                else:
                    data[child.tag] = [data[child.tag], child_data]
            else:
                data[child.tag] = child_data["text"]

        # 处理元素的文本和属性
        text = element.text.strip() if element.text is not None else ''
        data['text'] = text
        data.update(element.attrib)

        return data

    # 初始化一个空的列表来存储所有行的数据
    rows = []

    # 遍历XML的根元素的所有子元素
    for child in root:
        rows.append(parse_element(child))

    # 将所有行的数据转换为DataFrame
    df = pd.DataFrame(rows)

    return df


def df_to_xml(df):
    root = Element('root')

    # 遍历DataFrame并填充XML树
    for index, row in df.iterrows():
        element = SubElement(root, 'row')
        for column in df.columns:
            SubElement(element, column).text = str(row[column])

    # 生成XML字符串
    xml_string = tostring(root, encoding='unicode')
    return xml_string
    # 使用minidom格式化XML字符串
    # pretty_xml_string = minidom.parseString(xml_string).toprettyxml(indent="  ")
    #
    # return pretty_xml_string


if __name__ == '__main__':

    dir_path = "files"
    file_names = os.listdir(dir_path)
    print(file_names)
    files = [f for f in file_names if os.path.isfile(os.path.join(dir_path, f))]
    query = ""

    for index, file_name in enumerate(files[:5]):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        if file_name.endswith(".xml"):
            with open(file_path, "r", encoding="utf-8") as f:
                xml_data = f.read()
                df = xml_to_dataframe(xml_data)
                pprint(df.head())
                query += f"图表{index + 1}:\n" + df_to_xml(df) + "\n"

        elif file_name.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.loads(f.read())
                df = pd.json_normalize(json_data)
                print(df)
                query += f"图表{index + 1}:\n" + str(df.to_html()) + "\n"
        elif file_name.endswith(".csv"):
            df = pd.read_csv(file_path)
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="all")
            print(df)
            query += f"图表{index + 1}:\n" + df_to_xml(df) + "\n"

    # xml_string = df_to_xml(df)
    # print(xml_string)
    # print("-" * 70)
    prompt = """你是一个数据分析师，根据各种形式的数据进行分析、推断和总结。
我将给你多个HTML形式的表格，请你对各个表格中的数据进行分析，推断并总结表格所想要表达的信息或趋势，并在最后给出全局的总结和建议。
要求：
1.不需要将表格中的所有信息进行展示，可以根据情况适当阐述一些具有代表性的数据,确保这些数据的真实性。
2.适当描述数据的变化，并根据数据推断当前的状态或者趋势，但要严格使用图表中给出的数据，符合数据真实的变化趋势，不能捏造数据，虚构对数据的阐述。
3.对于数据的趋势推断，请严格根据数据变化判断，不能遗漏或者夸大说辞。
4.最后对根据所有图表的信息，生成一个总的概述总结。

以下给出各个图表：
{query}"""
    PromptTemplate = langchain.PromptTemplate(template=prompt, input_variables=["query"])
    chain = LLMChain(prompt=PromptTemplate, llm=llm, verbose=True)

    res = chain.invoke({"query": query})
    print(res["text"])

    prompt1 = """你是一个数据分析师，根据各种形式的数据进行分析、推断和总结。
我将给你多个HTML形式的表格，以及对表格的总结分析与建议，请你判断其是否存在错误的地方并更正
要求：
1.检查分析中的数据是否正确，如有不正确，请按照表格中的数据对分析结果进行改正。
2.对于分析中描述的变化趋势或者峰值节点，请你判断是否正确，如有不正确，请重新分析表格并改正。
3.对于数据的趋势推断，请严格根据数据变化判断，不能遗漏或者夸大说辞。
4.最后重新生成一份新的总结。

以下给出各个图表：""" + str(query) + """
以下是表格的总结与分析：
{query}
"""

    PromptTemplate = langchain.PromptTemplate(template=prompt1, input_variables=["query"])
    chain1 = langchain.LLMChain(prompt=PromptTemplate, llm=llm, verbose=True)

    res = chain1.invoke(res["text"])
    print(res["text"])
