# Leakguard
# [Conceptualization] Leakguard: LLM PII Leakage Diagnostic System

**Student Info**
- **Student No:** 22313531
- **Name:** 전정웅
- **E-Mail:** jjy4260@yu.ac.kr
- **address:** https://github.com/jjw4260/Leakguard/edit/main/Conceptualization_%5B22313531_%EC%A0%84%EC%A0%95%EC%9B%85%5D.md
---

## 1. Business Purpose
최근 대규모 언어 모델(Large Language Model, LLM)은 현대 사회의 핵심적인 인프라로 자리 잡았으나, 학습 과정에서 포함된 민감한 개인정보(Personal Identifiable Information, PII)를 모델이 기억하고 이를 외부로 유출할 수 있다는 심각한 보안 결함이 발견되고 있다. 기존의 보안 해결책들은 이러한 유출 위험을 수치상으로만 제시할 뿐, 실제 어떤 구체적인 정보가 어떻게 유출되는지에 대한 실질적인 위협을 증명하고 관리하는 데 한계가 있다.

기존 보안 체계의 사각지대를 해소하고, 모델의 안전성을 실질적으로 검증하기 위해 **Leakguard**의 개발이 필요하다. 본 프로그램을 통해 이루고자 하는 목표는 다음과 같다.

* **효율적인 보안 진단 및 관리:** 보안 취약점이 발견되더라도 이를 관리할 인력이 부족하거나 정보 공유가 원활하지 않으면 대응이 어렵다. Leakguard는 게시판 기능을 통한 공지사항 전달과 질의응답 기능을 제공하여, 보안 전문가들이 모델의 취약점 상태를 정확하게 공유하고 효율적으로 관리할 수 있도록 돕는다.
* **청렴하고 투명한 보안 이력 관리:** 모델에 대한 공격 시도와 그로 인한 유출 이력이 제대로 관리되지 않으면, 나중에는 어떤 경로로 정보가 새 나갔는지 파악하기가 불가능해진다. Leakguard는 서버에 모든 진단 이력과 유출 데이터를 저장하고 관리하게 함으로써, 데이터 유출 사고 발생 시 원인을 명확히 분석하고 정확한 사후 관리를 할 수 있도록 지원한다.
* **타겟 고객:** 개인정보 보호가 필수적인 정부 기관, 연구소, 대학교 내 보안 연구실이 주요 타겟이다. 특히 학생 정보를 다루는 대학교 동아리 관리 시스템이나 교육 기관처럼 출석 및 회원 관리가 필요한 곳에서 모델 도입 전 보안성을 확인하는 용도로 적합하다. 또한 자금 관리나 일정 공지 등 협업 기능이 포함되어 있어 중소규모 기업 및 개인 개발자들에게도 유용하다.

결과적으로 Leakguard는 고성능 모델을 실무에 도입하기 전 반드시 거쳐야 할 보안 가이드라인을 제시하며, 개인정보 유출로 인한 법적·윤리적 리스크를 사전에 차단하여 안전한 인공지능 활용 생태계를 조성하는 것을 최종 목적으로 한다.

---

## 2. System Context Diagram
*(이 섹션에는 시스템, 사용자, 외부 LLM API 간의 상호작용을 나타내는 다이어그램이 포함됩니다.)*

---

## 3. Use Case List

| Use Case | Actor | Description |
| :--- | :--- | :--- |
| **1) Register Assessment Target** | Member | 보안 점검이 필요한 특정 인물이나 데이터 범위를 시스템에 등록한다. 모델이 암기하고 있을 법한 파편화된 정보를 입력하여 정밀 공격 시드를 구축한다. |
| **2) Secure Authentication** | Member, Admin | 인가된 사용자만 접근 가능하도록 한다. 권한에 따라 조회 가능한 유출 데이터 범위와 진단 실행 권한을 엄격히 분리한다. |
| **3) Personal Leakage Monitoring** | Member | 일반 사용자는 본인 민감 정보가 타겟 모델로부터 어느 정도 확신도로 유출되는지 실시간 모니터링하며, 유출 발생 프롬프트 조합을 시각적으로 확인한다. |
| **4) Resource & Token Audit** | Member | 진단 과정에서 소모되는 GPU 자원 및 API 토큰 비용을 투명하게 확인하여 횡령이나 낭비를 방지한다. |
| **5) Execute Adaptive Assessment** | Admin | 관리자는 섀도우 모델을 구동하여 타겟 모델에 최적화된 공격 시나리오를 실행한다. 10분 내외의 정밀 스캔 데이터는 실시간으로 서버에 기록된다. |
| **6) Vulnerability Data Management** | Admin | 유출 데이터를 카테고리별로 분류하여 저장한다. 발견된 결함을 입력하면 시스템 보안 데이터베이스가 즉시 업데이트된다. |
| **7) Export Forensic Report** | Admin | 서버에 누적된 유출 증거와 진단 이력을 Excel이나 PDF 형식으로 추출하여 모델 안전성 입증 또는 보안 심사 자료로 사용한다. |
| **8) Strategic Schedule Planning** | Admin | 모델 업데이트 주기나 연구실 정책에 맞춰 정기 점검 스케줄을 계획한다. 달력 인터페이스로 구성원이 점검일을 사전에 인지하도록 돕는다. |

---

## 4. Concept of Operation

1.  **Register Assessment Target**
    * **Purpose:** 보안 점검이 필요한 특정 데이터 범위를 명확히 규정.
    * **Approach:** 신규 타겟에 대해 이름, 소속 등 모델 암기 가능성이 높은 시드 데이터를 입력하여 고유 ID 생성.
    * **Dynamics:** 필수 선행 단계로, 중복 등록 방지를 위해 고유 식별자 기반 DB 저장.
    * **Goals:** 체계적 타겟 관리를 통한 반복적 보안 스캔 기초 마련.

2.  **Shadow Pre-critique**
    * **Purpose:** 타겟 모델 타격 전, 공격 프롬프트의 유효성 사전 필터링.
    * **Approach:** 섀도우 모델들이 시드 데이터를 1차 비평하여 높은 등급을 받은 고순도 데이터만 선별.
    * **Dynamics:** 외부 API 호출 횟수 감소를 통한 비용 절감 및 실제 암기 신호 정밀 포착.
    * **Goals:** 노이즈 섞인 가짜 유출 배제 및 실질적 위협 데이터 추출.

3.  **Adaptive Security Probing**
    * **Purpose:** 타겟 모델 방어 체계 우회 및 실제 개인정보 유출 확인.
    * **Approach:** 승인된 데이터를 바탕으로 적응형 쿼리를 구성하여 전송. 10분 내외 집중 스캔을 통해 비평 토큰 응답 수집.
    * **Dynamics:** 관리자가 출석 체크를 하듯 편리하게 유출 여부를 확인하고 비고란에 메모 남김 가능.
    * **Goals:** 블랙박스 제약 극복 및 타겟 모델 내부의 암기된 PII 인퍼런스 성공.

4.  **Forensic Audit & Verification**
    * **Purpose:** 유출 데이터 진위성 최종 확정 및 이력 관리.
    * **Approach:** 타겟 모델 응답과 섀도우 모델 비평 결과를 교차 검증하여 최종 저장.
    * **Dynamics:** 저장과 동시에 총 유출 건수 및 보안 위험도 시각화.
    * **Goals:** 보안 이력의 투명한 관리 및 향후 모델 업데이트의 객관적 지표 활용.

5.  **Evidence Export**
    * **Purpose:** 분석된 보안 현황의 문서화.
    * **Approach:** 유출 리스트와 진단 이력을 Excel/메모장 형식으로 변환하여 로컬 폴더 생성.
    * **Dynamics:** 관리자 요청 시 특정 시점까지의 데이터 일괄 추출.
    * **Goals:** 외부 보안 심사 대응 및 연구 보고서 작성을 위한 공식 증거 확보.

---

## 5. Problem Statement

### 기술적 난제
* **섀도우 모델의 연산 자원 확보:** 복수의 섀도우 모델 구동을 위해 막대한 GPU 메모리 및 연산 성능 요구. 자원 부족 시 진단 속도 저하 문제 발생.
* **블랙박스 제약 조건:** 내부 파라미터를 알 수 없는 상태에서 API 응답과 비평 토큰만으로 유출을 증명해야 하므로 공격 프롬프트 최적화에 시행착오 발생 가능.
* **데이터 일관성 유지:** 파일 입출력 방식 사용 시 다수 사용자 동시 접근에 따른 데이터 무결성 유지 및 동기화의 어려움.

### 비기능적 요구사항
* **보안성:** 모든 진단 이력과 유출 PII 데이터는 암호화 저장 및 인가된 관리자 외 접근 철저 차단.
* **신뢰성:** 네트워크 장애로 API 응답이 끊겨도 이전까지의 진단 데이터 유실 방지를 위한 자동 저장 기능 지원.
* **효율성:** 섀도우 모델 사전 필터링으로 타겟 API 호출 횟수 최소화 및 비용/시간 최적화.
* **사용성:** 비전문가도 직관적으로 위험도를 파악할 수 있는 대시보드 형태 시각화 제공.
* **이식성:** Python 기반 GUI 제공을 통해 Windows 및 Linux 환경에서 독립적 실행 가능.

---

## 6. Glossary

| 용어 | 정의 및 설명 |
| :--- | :--- |
| **PII** | 이름, 학번, 이메일 등 특정 개인을 식별할 수 있는 민감 정보 데이터. |
| **Shadow Model** | 타겟 모델의 취약점을 시뮬레이션하고 공격 프롬프트 유효성을 검증하는 로컬 대조군 모델. |
| **MIA** | 특정 데이터가 모델 학습 과정에 포함되었는지 통계적으로 추론하여 프라이버시 침해를 측정하는 기법. |
| **Verified Verbatim** | 환각(Hallucination)이 아니라 실제 학습 데이터를 그대로 출력한 것으로 확인된 보안 결함 데이터. |
| **Critique Token** | 모델이 생성한 정보의 확신도나 정확도를 판단하여 출력하는 지표. |
| **Adversarial Trigger** | 암기 신호 편차를 극대화하여 PII 유출을 유도하기 위해 최적화된 적대적 공격 프롬프트. |
| **Audit History** | 공격 시도, 유출 결과, 토큰 소모량 등을 기록하여 서버에 저장한 데이터베이스. |
| **Inference** | 블랙박스 모델 쿼리를 통해 내부에 암기된 정보를 단계적으로 끌어내어 확인하는 과정. |

---

## 7. References
1. Yao, Y., Zhang, X., et al. “A Survey on Large Language Model Security and Privacy.” Patterns, 2024.
2. Cheng, S., Li, Y., et al. “Understanding PII Leakage in Large Language Models.” IJCAI, 2025.
3. “Evaluating Privacy Leakage and Memorization Attacks on Large Language Models.” Scientific Research Publishing, 2024.
4. “Sensitive Data Extraction from Black Box Large Language Models: Attack Vectors and Defenses.” TechRxiv preprint, 2024.
5. Lu, X., et al. “Do LLMs Really Memorize Personally Identifiable Information?” arXiv, 2026.
6. “Membership Inference Attacks on Tokenizers of Large Language Models.” arXiv, 2025.
7. “Membership Inference Attack Against Large Language Models.” arXiv, 2025.
8. Yin, L., et al. “LeakGuard: Detecting Memory Leaks Accurately and Scalably.” arXiv, 2025.
