### 모델 추론시 job Queue의 형식(FIFO/swtich) swap 옵션 적용
#### 현재 설정 현황
1. 1종류의 모델 다수를 inference 하면 절반씩나누어 HIGH Stream, low Stream에 할당되고
2. Priority Queue는 높은 Priority number가 높은 우선순위를 가진다.
3. FIFOQ 사용시 모든 Stream은 low Stream, 모든 Priority 값은 0 으로 설정된다
