import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import langchain
import pandas as pd
import torch
import yaml
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from lxml.etree import SubElement, Element
from lxml.html import tostring

os.environ["DASHSCOPE_API_KEY"] = "sk-146d6977be0b406fb18a4bb9c54d9cf0"
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatTongyi()



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

    for file_name in files[:1]:
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        df = ""
        if file_name.endswith(".xml"):
            with open(file_path, "r", encoding="utf-8") as f:
                xml_data = f.read()
                df = xml_to_dataframe(xml_data)
                print(df)

        elif file_name.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.loads(f.read())
                df = pd.json_normalize(json_data)
                print(df)
        elif file_name.endswith(".csv"):
            df = pd.read_csv(file_path)
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="all")
            print(df)

        xml_string = df_to_xml(df)
        print(xml_string)
        print("-" * 70)

        prompt = """你是一个数据分析师，根据各种心事的数据进行分析、推断和总结
我将给你一个xml形式的表格，请你对表格中的数据进行分析和总结，推断表格所想要表达的信息，以及趋势，并给出建议
表格为：
{query}"""
        PromptTemplate = langchain.PromptTemplate(template=prompt, input_variables=["query"])
        chain = llm | PromptTemplate | {"query": RunnablePassthrough()} | StrOutputParser()

        res = chain.invoke(xml_string)
        print(res)

    # 转换XML数据为DataFrame
    # df = xml_to_dataframe(xml_data)

    # 显示DataFrame
    # print(df)
