# -*- coding: utf-8 -*-
"""
Pydantic 数据模型：统一试卷、作答、判分、台账以及 HTTP 接口的数据结构。
既用于 FastAPI 入参/出参校验，也用于 Agent 工具与评估脚本，保证全链路结构一致。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

QuestionType = Literal["单选题", "多选题", "判断题", "简答题"]
EduLevel = Literal["公司级", "项目级", "班组级"]


# ============================ 试卷 ============================
class Question(BaseModel):
    """单道试题。"""

    index: int = Field(..., description="题号，从 1 开始")
    q_type: QuestionType = Field(..., description="题型")
    content: str = Field(..., description="题干")
    options: list[str] = Field(default_factory=list, description="选项；判断题/简答题可为空")
    answer: str = Field("", description="参考答案")
    analysis: str = Field("", description="答案解析")
    score: float = Field(default=5.0, description="该题满分")


class Exam(BaseModel):
    """一份完整考核试卷。"""

    title: str
    craft_type: str = Field(..., description="工种")
    edu_level: str = Field(..., description="三级教育层级")
    questions: list[Question]
    total_score: float = 100.0
    references: list[str] = Field(default_factory=list, description="参考资料来源")


# ============================ 作答与判分 ============================
class AnsweredQuestion(BaseModel):
    """考生作答。"""

    index: int
    q_type: QuestionType
    student_answer: str = Field("", description="考生答案；选择题填选项字母，判断题填 对/错")


class SubmittedPaper(BaseModel):
    """考生提交的整份试卷。"""

    craft_type: str = ""
    edu_level: str = ""
    trainee_name: str = ""
    answers: list[AnsweredQuestion]


class GradedQuestion(BaseModel):
    """单题判分结果。"""

    index: int
    q_type: QuestionType
    standard_answer: str
    student_answer: str
    full_score: float
    score: float
    correct: bool
    comment: str = ""


class GradeResult(BaseModel):
    """整份试卷判分结果。"""

    trainee_name: str = ""
    total_score: float = 0.0
    full_score: float = 100.0
    objective_score: float = 0.0
    subjective_score: float = 0.0
    passed: bool = False
    pass_line: float = 60.0
    details: list[GradedQuestion] = Field(default_factory=list)


# ============================ 培训资料 ============================
class MaterialSection(BaseModel):
    """培训资料章节。"""

    heading: str = Field(..., description="章节标题")
    content: str = Field("", description="章节内容（支持多段落，换行分隔）")


class TrainingMaterial(BaseModel):
    """按工种 + 三级教育层级生成的标准化培训资料。"""

    title: str
    craft_type: str = Field(..., description="工种")
    edu_level: str = Field(..., description="三级教育层级")
    summary: str = Field("", description="培训资料概述/适用范围")
    sections: list[MaterialSection] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list, description="参考资料来源")


# ============================ 台账 ============================
class TrainingLedger(BaseModel):
    """安全培训台账（合规留痕）。"""

    craft_type: str
    edu_level: str
    training_topic: str = ""
    trainer: str = ""
    location: str = ""
    training_date: str = ""
    duration_hours: float = 2.0
    trainee_names: list[str] = Field(default_factory=list)
    trainee_count: int = 0
    average_score: float = 0.0
    pass_count: int = 0
    grade_summary: str = ""
    grade_results: list[GradeResult] = Field(default_factory=list)
    remark: str = "本台账由安全培训智能 Agent 生成初稿，需安全员复核签字后归档。"


# ============================ HTTP 接口模型 ============================
class AgentChatRequest(BaseModel):
    query: str = Field(..., description="用户自然语言指令")
    thread_id: str = Field(default="default", description="会话 ID，用于多轮记忆")
    craft_type: Optional[str] = None
    edu_level: Optional[str] = None


class ExamRequest(BaseModel):
    craft_type: str
    edu_level: str
    question_counts: Optional[dict[str, int]] = Field(
        default=None, description="各题型数量，如 {'单选题':5,'多选题':3,'判断题':3,'简答题':2}"
    )


class GradeRequest(BaseModel):
    exam: Exam
    paper: SubmittedPaper


class LedgerRequest(BaseModel):
    craft_type: str
    edu_level: str
    trainee_names: list[str] = Field(default_factory=list)
    grade_results: list[GradeResult] = Field(default_factory=list)
    trainer: str = ""
    location: str = ""
    training_date: str = ""
    training_topic: str = ""


class StandardResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Optional[dict] = None
