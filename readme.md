# da
 - Python::Miniconda 기반해 몇 가지 거시경제 지표와 금융상품 가격의 관계를 확인하는 스크립트예요
 - 대시보드: http://114.204.56.226:11115/

# 설명
 - 초회 실행시 f.csv가 필요해요
 - Inflation Index (호가 deflator)는 매일 t10yie와 전월 CPI로 계산합니다. t10yie는 "(US10Y Quote) - (Inflation-indexed US10Y investment-based Quote)"로 daily 계산 가능해요
 - Prominent 금융상품에 대해 rolling over가 고려되지 않은 중대한 문제가 있어요

# 계획
 - 대시보드::Macro::중요 후행성 거시지표 및 연관 계열 표시 추가

# 참고문헌
 - Bernanke's Lecture Serie

# 라이센스
 - 스크립트는 wrapper라서 아무렇게나 써도 되요
