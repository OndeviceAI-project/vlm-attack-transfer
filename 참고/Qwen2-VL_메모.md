# Qwen2-VL-2B 메모 (B팀 참고, 규칙 아님)

미리 확인해 둔 것이다. 다른 방법으로 해도 "공격이 보는 사진 = 평가가 보는 사진"(차이 0)과 census만 통과하면 된다.

- **사진·토큰.** 프로세서를 `min_pixels=max_pixels=448*448`로 만들면 448×448 사진이 14px 조각 1024개 → 이미지 토큰 256개, 입력 284토큰, `image_grid_thw = [[1, 32, 32]]`. 크기를 고정하지 않으면 사진마다 토큰 수가 달라진다.
- **pixel_values 모양.** `(1024, 1176)`. 조각을 한 줄로 편 것이고 같은 사진을 2프레임처럼 복사한다(1176 = 3 × 2 × 14 × 14). 직접 만들 때(transformers 5.19): 정규화(프로세서의 `image_mean`·`image_std`) → `reshape(1, 3, 16, 2, 14, 16, 2, 14)` → `permute(0, 2, 5, 3, 6, 1, 4, 7)` → 2프레임 복사 → `reshape(1024, 1176)`. 프로세서 결과와 차이 0.
- **프로세서 출력.** `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `mm_token_type_ids`. 모델에 넣는 텐서들의 길이를 `input_ids`와 맞춘다.
- **종료 토큰 2개.** `<|im_end|>`, `<|endoftext|>`. 하나만 보면 끝난 답을 안 끝난 것으로 센다.
- **채팅 양식.** "You are a helpful assistant." 시스템 문장이 자동으로 붙는다. 공격과 평가가 같은 양식이면 된다.
- **부품 이름** (`settings.MODELS["qwen2vl"]["parts"]`): V `model.visual.blocks`(Linear 약 128), C `model.visual.merger`(2), L `model.language_model`(약 196), H `lm_head`(1). census로 확인. `model.visual.patch_embed`는 Conv라 대상이 아니다. 2B는 입력 임베딩과 출력층이 가중치를 공유할 수 있는데(`tie_word_embeddings`), `apply_rtn`은 새 Parameter로 갈아 끼우므로 1차(V만)에서는 상관없다.
- **bnb 경우.** `BitsAndBytesConfig(..., llm_int8_skip_modules=["model.visual.merger", "model.language_model", "lm_head"])`. CUDA에서만 된다.
- **메모리·시간.** 공격 생성은 bf16(fp32 역전파는 메모리 부족). 모델 4.4GB, 16GB GPU면 될 것으로 보임. 안 되면 사진 336으로 줄이는 것을 제안. 공격 사진 1장 수 분~20분, 평가는 경우당 15분 안팎(공격 사진이 512토큰까지 가므로). bf16이 되는 GPU 필요(Colab 무료 T4는 안 됨).
- **패키지.** A팀 기계에서 확인된 조합: torch 2.11(cu128) / transformers 5.19 / bitsandbytes 0.50 / pillow 11.2.
