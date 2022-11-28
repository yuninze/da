# da
 - Python::Miniconda 기반해 몇 가지 거시경제 지표와 금융상품 가격의 관계를 확인하는 유틸리티 스크립트입니다

# 설명
 - Trading을 위한 스크립트가 아닙니다
 - 초회 실행시 f.csv가 필요합니다
 - Inflation Index (호가 deflator)는 매일 t10yie와 전월 CPI로 계산합니다. t10yie는 "(US10Y Quote) - (Inflation-indexed US10Y investment-based Quote)"로 daily 계산 가능합니다
 - 이외 인용하는 거시지표에 대해서는 참고문헌을 숙독하세요
 - Prominent 금융상품에 대해 rolling over가 고려되지 않은 중대한 문제가 있습니다

# 계획
 - rolling over 고려하기
 - 빈도 조절 (frequency manipulation, resampling)

# 참고문헌
 - Chairman Bernanke's Lecture Series (2012)
 - Fred
 - Fed Challenge

# 라이센스
 - 스크립트는 모두 wrapper이므로 아무렇게나 써도 됩니다
 - 일부 거시지표는 licensed이므로 citation 유의해야 합니다
