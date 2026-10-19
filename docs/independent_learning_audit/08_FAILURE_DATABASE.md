# 08. База найденных отказов и границы классификации

VERIFIED относится к наличию события или декларации результата. Оно не означает, что каждая предполагаемая причина доказана. В `failures.csv` сохранены все найденные `success=false` поддержанных семейств raw episodes, завершённые ранние диагностические отказы и отдельные инженерные случаи. Дубликат исторического события в двух источниках не считается новой попыткой. Число строк базы не является числом независимых экспериментов.

Для старых failed episodes без явного диагноза причина указана UNKNOWN. Аудит не переоценивал каждый контакт, не восстанавливал отсутствующий simulator state и не приписывал всем отказам последний диагноз. База не может включать недоступные или ранее удалённые артефакты. Историческое удаление отдельных диагностических записей по прежнему запросу пользователя отмечено в02; во время этого аудита данные не удалялись.

| Случай | Наблюдение | Причина и уверенность | Исправление и повторная проверка | Доказательство |
| --- | --- | --- | --- | --- |
| firmware_sync_write | Wrong assertion that factory sync-write was absent | Arm_Sync_Write existed; rc2 withdrawn before flashing (VERIFIED) | withdrawn; no proof of physical fix | `source/docs/firmware-rc2-withdrawal-20260929.md` |
| pipeline_scope | Ready wording preceded a complete mobile policy executor | Component readiness substituted for end-to-end acceptance (STRONGLY_SUPPORTED) | not qualified | `19_CONVERSATION_AND_DECISION_AUDIT.md#d04` |
| browser_stop | Cycle stopped about 46 seconds after leaving tab | Browser focus/visibility stop coupled to background job (VERIFIED) | 0293b06; background behaviour corrected | `02_FULL_HISTORY.md` |
| exposure_budget | 20000 updates with 129548 frames/batch16 = 2.47 passes | Too few passes relative to configured minimum8 (VERIFIED) | minimum exposure corrected; did not establish full success | `02_FULL_HISTORY.md` |
| unpaired_regression | 6/10 then 4/10 with changed scenes/initialization | Attribution to model/data change confounded by evaluation changes (VERIFIED) | paired gates added later | `19_CONVERSATION_AND_DECISION_AUDIT.md#d12` |
| maintenance_pause | Installer maintenance pause interpreted as user pause | Pause origin was not separated (VERIFIED) | 3fb6905; old cycle now permanently stopped | `19_CONVERSATION_AND_DECISION_AUDIT.md#d14` |
| neural_consumer_absent | Parameter trials did not train ACT | Architecture change dropped required dataset-to-neural-model consumer (VERIFIED) | 2161826/853c4c0 restore neural pipeline | `19_CONVERSATION_AND_DECISION_AUDIT.md#d15` |
| renderer_stall | 167 attempts unchanged for more than2h | Render path blocked worker (VERIFIED) | render execution repaired; no skill acceptance implied | `19_CONVERSATION_AND_DECISION_AUDIT.md#d17` |
| equal_score | 119047 updates, candidate3/10 reference3/10 | No measured improvement; exact learning cause UNKNOWN (VERIFIED) | reference retained; corrections collected | `evidence/run_inventory.json` |
| lineage | Interrupted same-ID copy shadowed exact training source | Episode lookup ignored training content hash (VERIFIED) | 3568b30 exact-byte resolution | `source/simulator/episode_lineage.py` |
| socket_path | Private service UNIX socket path too long | Deep frozen source path (VERIFIED) | 929497a short private vprobe root | `source/simulator/live_room_diagnostic.py` |
| frontier | Sock found, basket not confirmed after24 views | Cheap low-information turns dominated useful views (STRONGLY_SUPPORTED) | 934c479; later both objects found | `02_FULL_HISTORY.md` |
| resource_wait_budget | Hour expired before scene start | Resource waiting incorrectly consumed active execution budget (VERIFIED) | 8c6ce11 active-time-only budget | `source/simulator/live_room_diagnostic.py` |
| rim_aim | Basket survey wrist pose unreachable | Aimed at sidewall centre rather than framing observed rim (STRONGLY_SUPPORTED) | c746696; subsequent both-object confirmation | `source/simulator/surface_destination.py` |
| cad_aabb | Fresh geometric alignment did not converge before ACT | World AABB empty corner falsely intersected tilted full CAD mesh (VERIFIED) | a5d9ea7 separating-plane refinement; obstacle control still rejected | `02_FULL_HISTORY.md` |
| optical_range | Four close views refused before ACT | Fixed .34m optical range infeasible at .20m standoff (VERIFIED) | 726f8cb finite feasible distances with unchanged gates | `02_FULL_HISTORY.md` |
| cadence | Fresh unsafe shoulder/elbow output after200 actions | 12 manual ticks after10ACT produced22s physical versus10s actor clock; contribution not isolated (VERIFIED) | cfe4aa7 timing10/10; later grasp still failed | `evidence/historical/cadence-physics-regression.json` |
| reader_decode | Slow data load/updates | Action horizon decoded20 unused future RGB frames (VERIFIED) | 1000 to0 extra decodes in50 queries; targets byte-identical | `evidence/historical/projected-reader-benchmark.json` |
| memory_accounting | False resource wait after Docker closure | Bypassed native resource_budget; psutil and native availability differed (VERIFIED) | 09cb728 common native budget; thresholds not lowered | `source/simulator/live_room_diagnostic.py` |
| incomplete_checkpoint | Weights existed without complete optimizer state | Disk disconnect interrupted checkpoint save (VERIFIED) | safe resume from complete017000; incomplete data preserved | `checkpoints.csv` |
| refusal_695 | Fresh gripper48→75.8205 refused; no grasp/lift/place | Numerical mechanism verified; unique learning cause UNKNOWN (VERIFIED) | read-only replay exact; no repair or new physical rollout claimed | `examples/refusal-695/refusal_forensics.json` |
| state_time_shift | Shoulder123 exceeds training max116.5517; clock40.75 associated with retreat | Confirmed state/phase mismatch; causal proportion UNKNOWN (VERIFIED) | requires contract/recovery ablations before long training | `evidence/runtime_distribution_followup.json` |
| feedback_claim | Old raw replies were overinterpreted as current joint feedback | Historical interface confused with current command-only configuration (VERIFIED) | claim withdrawn after user correction | `19_CONVERSATION_AND_DECISION_AUDIT.md#d26` |

В базе 458 строк, из них 424 отдельных raw manifest с отрицательной декларацией. Частотный анализ всех симптомов не заменяет причинный разбор: одни и те же сообщения guard могут возникать по разным причинам.
