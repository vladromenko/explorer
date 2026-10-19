# 07. Teacher, управление и время

## 8. Текущая архитектура полного поведения: обученная часть и написанная часть

```text
Просьба пользователя
  → проверка поддержанного навыка и параметров сцены
  → RGB-D + семантический детектор + лидары/карта
  → выбор полезного обзора и проверяемого пути
  → поиск носка и корзины, подход, повторное наблюдение
  → свежая цель + согласованный phase/clock handoff
  → ACT управляет рукой, база удерживается неподвижной
  → наблюдаемая проверка удержания
  → алгоритмическая безопасная перевозка
  → свежий обзор корзины, ACT размещения
  → отпускание, отход, наблюдаемая проверка
  → независимая оценка истинного результата после действий
```

Поиск, карта, A*, вычисление полезности обзора, IK, CAD-проверки, координатор стадий и safety написаны кодом. Моторные предсказания ACT хранятся в обученных весах. Handoff использует статистику команд/времени из учебных фаз и наблюдаемую геометрию корзины, затем вызывает **свежий ACT inference**. Он не проигрывает сохранённую последовательность команд из одного эпизода. Тем не менее это дополнительная вручную заданная композиция, которую необходимо проверять как отдельный целый сценарий. [Координатор][coordinator], [room policy/handoff][room-policy].

Следовательно, результат будет **гибридной системой**, а не одной нейросетью, которая из произвольной фразы самостоятельно открыла всю физику, навигацию и новые навыки.


[act-external]: https://github.com/tonyzhaozh/act
[archive]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_archive.py
[audit-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING_AUDIT.md
[audit08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/CODE-AUDIT-20261008.ru.md
[bridge]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/lerobot_bridge.py
[c-0293b06]: https://github.com/vladromenko/explorer/commit/0293b068acf58542b8ec61605c75ab7feed8e802
[c-09cb728]: https://github.com/vladromenko/explorer/commit/09cb7282c2a9a52977329c6cff78470e9ebe1a3d
[c-0b231e4]: https://github.com/vladromenko/explorer/commit/0b231e48e2e7af1256d2a60a6cc67e6f1287556f
[c-0bd7728]: https://github.com/vladromenko/explorer/commit/0bd772860832ef54d8a65aed0d8abaa8f1bce0d7
[c-146a463]: https://github.com/vladromenko/explorer/commit/146a4630ef407efef09b5f6ec3944909b206634e
[c-1d6ddf5]: https://github.com/vladromenko/explorer/commit/1d6ddf538369e6c54b1fedb22f6e57ef2ec119d2
[c-1da8ecb]: https://github.com/vladromenko/explorer/commit/1da8ecb26efaa798d4293f126d7d123aff7985e5
[c-2161826]: https://github.com/vladromenko/explorer/commit/2161826271fb90efd9bd6e9ae7cc818590d09e51
[c-2d3aeae]: https://github.com/vladromenko/explorer/commit/2d3aeae87d190f83449981fbdad5cf5e55673f36
[c-2e484e3]: https://github.com/vladromenko/explorer/commit/2e484e39f98078d99b5e579c043bc1f6e5072ee6
[c-30271ba]: https://github.com/vladromenko/explorer/commit/30271bae7e0c53a3a06462fa55a7da962dace506
[c-3568b30]: https://github.com/vladromenko/explorer/commit/3568b300116926717f03b8b13cf01c3984702093
[c-37ae853]: https://github.com/vladromenko/explorer/commit/37ae853480c6dc88c7dbbebdfd348e38fa0876f2
[c-3fb6905]: https://github.com/vladromenko/explorer/commit/3fb69053bafc5185b70dcbef75988bf79d043bb2
[c-4146b1c]: https://github.com/vladromenko/explorer/commit/4146b1c6ba509b31431fcbde5daa291d184b869e
[c-42374fe]: https://github.com/vladromenko/explorer/commit/42374fe013127d9a3d82eb0a81011f51c29e05c8
[c-43304eb]: https://github.com/vladromenko/explorer/commit/43304eb10133002fe2fe11395d9f2d67317d7c16
[c-4386cc0]: https://github.com/vladromenko/explorer/commit/4386cc0cdf162de3dbaab6d3568335301146003d
[c-5b27e21]: https://github.com/vladromenko/explorer/commit/5b27e215b94ff0d483806391559f4dc83a8b7576
[c-5bb6914]: https://github.com/vladromenko/explorer/commit/5bb69148a733272d53f3897acbcab65a6a335f4b
[c-5db446d]: https://github.com/vladromenko/explorer/commit/5db446d32752698256ddb7e4a5c36cc258967605
[c-65c5738]: https://github.com/vladromenko/explorer/commit/65c57387e988998cc07349d5fc89e624dcb9d122
[c-67908f9]: https://github.com/vladromenko/explorer/commit/67908f956eea5b5f72db5c906bf989d8810c2504
[c-6d6394a]: https://github.com/vladromenko/explorer/commit/6d6394a4ba3a2e92381b2f45fbd60f4867e572b5
[c-726f8cb]: https://github.com/vladromenko/explorer/commit/726f8cbd72f7d4723d8a0e8500099efbae3dfb67
[c-853c4c0]: https://github.com/vladromenko/explorer/commit/853c4c0d9c418fac3a4b9f1318c9b036e3c11b33
[c-8c6ce11]: https://github.com/vladromenko/explorer/commit/8c6ce11c8f27c91e60aaa8c9623a55040a019706
[c-929497a]: https://github.com/vladromenko/explorer/commit/929497a7f85366983d2bfbdf4dea9910e727ced2
[c-934c479]: https://github.com/vladromenko/explorer/commit/934c47986e11621fe0e0a074e51866e2bc81b92b
[c-a5d9ea7]: https://github.com/vladromenko/explorer/commit/a5d9ea7c3f40324fe4c37a54bd3425b46bf34f65
[c-b4186e1]: https://github.com/vladromenko/explorer/commit/b4186e1749ad2ac12d845fc51e63adf74828fad8
[c-b6373e3]: https://github.com/vladromenko/explorer/commit/b6373e35bd827ba42f50d5d1b2e6954e575ed61e
[c-b828129]: https://github.com/vladromenko/explorer/commit/b8281297e4eb3f846bba22193710b04cacc6775a
[c-b9cd5e3]: https://github.com/vladromenko/explorer/commit/b9cd5e31f29d86935c60e24ff577878eaa785801
[c-bf36e4e]: https://github.com/vladromenko/explorer/commit/bf36e4ea370d44eb1757c093247bb1e10e167ff6
[c-c553017]: https://github.com/vladromenko/explorer/commit/c55301788dca39584bcf7d68f761580de2a6c7ac
[c-c746696]: https://github.com/vladromenko/explorer/commit/c7466967cbe00f26ff7ee867cd0ff8542d8b0050
[c-cfbd8eb]: https://github.com/vladromenko/explorer/commit/cfbd8eb0d2f6f0b4defa055aac9ea972380f6c84
[c-cfe4aa7]: https://github.com/vladromenko/explorer/commit/cfe4aa7044e75798384d8fd6269fcab84761820d
[c-d1c4d11]: https://github.com/vladromenko/explorer/commit/d1c4d1155fad093f5ca04c92343dd1c0d44a6fb5
[c-e9ad08b]: https://github.com/vladromenko/explorer/commit/e9ad08be2d3694369610d88676024015b21ec6d8
[c-f2ac13d]: https://github.com/vladromenko/explorer/commit/f2ac13dff1b378fc05600bd1443fb537441d289e
[c-fd62906]: https://github.com/vladromenko/explorer/commit/fd629068482570e18bc5d2d087133dc647f8a957
[cad-external]: https://github.com/dimwael/yahboom_m3pro_description/tree/bdac682e57a8eeaf7b18eece44bfbfc54de98e8a
[contract]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_policy_contract.py
[coordinator]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/room_cleanup.py
[dagger-external]: https://arxiv.org/abs/1011.0686
[diagnostic]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/live_room_diagnostic.py
[diffusion-external]: https://github.com/real-stanford/diffusion_policy
[early-learning]: https://github.com/vladromenko/explorer/blob/2d3aeae87d190f83449981fbdad5cf5e55673f36/docs/LEARNING-IMPLEMENTATION.ru.md
[entry]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/train_entry.py
[environment]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/environment.py
[eval-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EVALUATION_REPORT.md
[explorer-model]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/explorer_model.py
[guide]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING-GUIDE.ru.md
[implementation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/LEARNING-IMPLEMENTATION.ru.md
[intervention]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_intervention.py
[learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/learning.py
[lerobot-external]: https://huggingface.co/docs/lerobot/en/act
[lineage]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/episode_lineage.py
[local-grasp]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/local_grasp_experiment.py
[mjx-external]: https://mujoco.readthedocs.io/en/stable/mjx.html
[model-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/MODEL_AND_TRAINING.md
[neural]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/neural_recovery.py
[observer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/checkpoint_monitor.py
[onboard]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EXPLORER-ONBOARD-RELEASE-20261010.ru.md
[origin]: https://github.com/vladromenko/explorer
[phase]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/phase_learning.py
[r-adaptive17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/adaptive-learning-20261017.json
[r-audit12]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-audit-e7905c76974c-20261012.json
[r-auto16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-20261016.json
[r-causal17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/causal-repair-20261017.json
[r-correction16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-correction-20261016.json
[r-corrective15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-corrective-cycle-20261015.json
[r-hierarchy15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/hierarchical-rework-20261015.json
[r-jetson14]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-training-d6ebe13cae23-jetson-20261014.json
[r-macos]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-macos-20261011.json
[r-monitor17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-causal-monitor-20261017.json
[r-neural18]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/neural-recovery-20261018.json
[r-overnight]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-overnight-check-20261012.json
[record]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_demonstrations.py
[repair]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/failure_repair.py
[repo]: https://github.com/vladromenko/explorer-learning
[room-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/ROOM-CLEANUP.ru.md
[room-learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_learning.py
[room-policy]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_policy.py
[room-port]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_cleanup_port.py
[room08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/audits/20261008-room-resume.md
[runbook]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/RUNBOOK.md
[search]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/skill_training.py
[simopt-external]: https://arxiv.org/abs/1810.05687
[smolvla-external]: https://huggingface.co/docs/lerobot/en/smolvla
[stack]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/learning_stack.py
[transfer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_transfer.py
[unified]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/UNIFIED-AUTONOMY.ru.md
[validation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_validation.py
[vision]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/OBJECT-VISION.ru.md
## 9. Полный программный учебный цикл

1. **Фиксация задачи/контракта.** Сохраняются конфигурация сцены, разрешённые наблюдения, версия кода, геометрия, random seed и критерии.
2. **Физические показы в симуляторе.** Teacher выполняет действия через физику; записываются pre-action RGB/состояния, команды, фазы и происхождение. Удачи и неудачи сохраняются отдельно.
3. **Диагноз после остановки.** Восприятие/путь/IK/инфраструктура отделяются от неудачи моторной политики. Дополнительные эпохи не должны служить универсальным ремонтом.
4. **Корректирующие примеры.** Для поддержанного класса ошибки ученик выполняет ограниченный префикс, затем teacher пытается исправить состояние. Student-команды маркируются `training_target_valid=false`; целями служат экспертные корректировки.
5. **Порог качества нового материала.** Для нового полного room collector нужны минимум восемь полных успешных показов; для pickup-коррекций — минимум четыре действительно успешных исправления посещённых студентом состояний с исполненным префиксом и реальным движением expert. Неподвижный безопасный выход не считается полезной коррекцией.
6. **Совместимый датасет с историей.** Проверяются SHA, contract, camera/geometry, timestamps, alignment, маски. Нельзя считать все старые записи автоматически совместимыми. Успешные совместимые данные объединяются; старые файлы не стираются.
7. **Разделение целыми эпизодами/seed.** Сохраняется inherited validation; финальные сцены отделены от train и подбора.
8. **ACT.** Один learner, warm start или полноценный resume в зависимости от состояния; минимум восемь проходов текущего выбранного набора; checkpoint и optimizer/RNG сохраняются.
9. **Промежуточная диагностика.** Один наблюдатель стабильных числовых checkpoint, без чтения незавершённой записи. После resume минимум 100 новых обновлений; затем ограничения по времени/числу новых шагов и ресурсам. Это способ рано обнаруживать ошибки интерфейса и поведения, не финальная приёмка.
10. **Выбор модели.** Reference и shortlisted checkpoints проверяются на одинаковых десяти новых сценах; низкий offline MAE сам по себе не продвигает модель. При равенстве сохраняется reference.
11. **Следующий диагноз / ограниченный раунд.** Если поддержанная причина допускает коррекции, программа собирает их и дообучает. Локальный recovery ограничен тремя корректирующими раундами; room-cycle также конечен. При неизвестном программном дефекте выставляется `needs_algorithm_repair` / `needs_room_algorithm_repair`, сохраняются доказательства. Инженер должен устранить новую причину, а не запустить идентичную очередь на сутки.
12. **Финальная независимая оценка полного сценария.** Две группы по 50 заранее зафиксированных новых сцен, 100/100 полных успехов, ноль столкновений, без teacher fallback. Неудачные финальные сцены остаются исключёнными из будущего обучения; следующий допуск требует новых holdout-сцен.
13. **Пакет и архив.** Только после принятия — проверка кода/весов/контрактов/архивов, неактивный Jetson пакет, отчёты и точные SHA. `physical_ready=false` до отдельного реального испытания.

Код локального и комнатного циклов: [neural_recovery.py][neural], [room_learning.py][room-learning], [room_intervention.py][intervention]. Наличие этого кода не означает, что все будущие стадии уже успешно выполнены. В частности, новый полный генератор исправлений ещё не доказал нужное число успешных полных исправлений на текущей миссии.


[act-external]: https://github.com/tonyzhaozh/act
[archive]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_archive.py
[audit-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING_AUDIT.md
[audit08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/CODE-AUDIT-20261008.ru.md
[bridge]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/lerobot_bridge.py
[c-0293b06]: https://github.com/vladromenko/explorer/commit/0293b068acf58542b8ec61605c75ab7feed8e802
[c-09cb728]: https://github.com/vladromenko/explorer/commit/09cb7282c2a9a52977329c6cff78470e9ebe1a3d
[c-0b231e4]: https://github.com/vladromenko/explorer/commit/0b231e48e2e7af1256d2a60a6cc67e6f1287556f
[c-0bd7728]: https://github.com/vladromenko/explorer/commit/0bd772860832ef54d8a65aed0d8abaa8f1bce0d7
[c-146a463]: https://github.com/vladromenko/explorer/commit/146a4630ef407efef09b5f6ec3944909b206634e
[c-1d6ddf5]: https://github.com/vladromenko/explorer/commit/1d6ddf538369e6c54b1fedb22f6e57ef2ec119d2
[c-1da8ecb]: https://github.com/vladromenko/explorer/commit/1da8ecb26efaa798d4293f126d7d123aff7985e5
[c-2161826]: https://github.com/vladromenko/explorer/commit/2161826271fb90efd9bd6e9ae7cc818590d09e51
[c-2d3aeae]: https://github.com/vladromenko/explorer/commit/2d3aeae87d190f83449981fbdad5cf5e55673f36
[c-2e484e3]: https://github.com/vladromenko/explorer/commit/2e484e39f98078d99b5e579c043bc1f6e5072ee6
[c-30271ba]: https://github.com/vladromenko/explorer/commit/30271bae7e0c53a3a06462fa55a7da962dace506
[c-3568b30]: https://github.com/vladromenko/explorer/commit/3568b300116926717f03b8b13cf01c3984702093
[c-37ae853]: https://github.com/vladromenko/explorer/commit/37ae853480c6dc88c7dbbebdfd348e38fa0876f2
[c-3fb6905]: https://github.com/vladromenko/explorer/commit/3fb69053bafc5185b70dcbef75988bf79d043bb2
[c-4146b1c]: https://github.com/vladromenko/explorer/commit/4146b1c6ba509b31431fcbde5daa291d184b869e
[c-42374fe]: https://github.com/vladromenko/explorer/commit/42374fe013127d9a3d82eb0a81011f51c29e05c8
[c-43304eb]: https://github.com/vladromenko/explorer/commit/43304eb10133002fe2fe11395d9f2d67317d7c16
[c-4386cc0]: https://github.com/vladromenko/explorer/commit/4386cc0cdf162de3dbaab6d3568335301146003d
[c-5b27e21]: https://github.com/vladromenko/explorer/commit/5b27e215b94ff0d483806391559f4dc83a8b7576
[c-5bb6914]: https://github.com/vladromenko/explorer/commit/5bb69148a733272d53f3897acbcab65a6a335f4b
[c-5db446d]: https://github.com/vladromenko/explorer/commit/5db446d32752698256ddb7e4a5c36cc258967605
[c-65c5738]: https://github.com/vladromenko/explorer/commit/65c57387e988998cc07349d5fc89e624dcb9d122
[c-67908f9]: https://github.com/vladromenko/explorer/commit/67908f956eea5b5f72db5c906bf989d8810c2504
[c-6d6394a]: https://github.com/vladromenko/explorer/commit/6d6394a4ba3a2e92381b2f45fbd60f4867e572b5
[c-726f8cb]: https://github.com/vladromenko/explorer/commit/726f8cbd72f7d4723d8a0e8500099efbae3dfb67
[c-853c4c0]: https://github.com/vladromenko/explorer/commit/853c4c0d9c418fac3a4b9f1318c9b036e3c11b33
[c-8c6ce11]: https://github.com/vladromenko/explorer/commit/8c6ce11c8f27c91e60aaa8c9623a55040a019706
[c-929497a]: https://github.com/vladromenko/explorer/commit/929497a7f85366983d2bfbdf4dea9910e727ced2
[c-934c479]: https://github.com/vladromenko/explorer/commit/934c47986e11621fe0e0a074e51866e2bc81b92b
[c-a5d9ea7]: https://github.com/vladromenko/explorer/commit/a5d9ea7c3f40324fe4c37a54bd3425b46bf34f65
[c-b4186e1]: https://github.com/vladromenko/explorer/commit/b4186e1749ad2ac12d845fc51e63adf74828fad8
[c-b6373e3]: https://github.com/vladromenko/explorer/commit/b6373e35bd827ba42f50d5d1b2e6954e575ed61e
[c-b828129]: https://github.com/vladromenko/explorer/commit/b8281297e4eb3f846bba22193710b04cacc6775a
[c-b9cd5e3]: https://github.com/vladromenko/explorer/commit/b9cd5e31f29d86935c60e24ff577878eaa785801
[c-bf36e4e]: https://github.com/vladromenko/explorer/commit/bf36e4ea370d44eb1757c093247bb1e10e167ff6
[c-c553017]: https://github.com/vladromenko/explorer/commit/c55301788dca39584bcf7d68f761580de2a6c7ac
[c-c746696]: https://github.com/vladromenko/explorer/commit/c7466967cbe00f26ff7ee867cd0ff8542d8b0050
[c-cfbd8eb]: https://github.com/vladromenko/explorer/commit/cfbd8eb0d2f6f0b4defa055aac9ea972380f6c84
[c-cfe4aa7]: https://github.com/vladromenko/explorer/commit/cfe4aa7044e75798384d8fd6269fcab84761820d
[c-d1c4d11]: https://github.com/vladromenko/explorer/commit/d1c4d1155fad093f5ca04c92343dd1c0d44a6fb5
[c-e9ad08b]: https://github.com/vladromenko/explorer/commit/e9ad08be2d3694369610d88676024015b21ec6d8
[c-f2ac13d]: https://github.com/vladromenko/explorer/commit/f2ac13dff1b378fc05600bd1443fb537441d289e
[c-fd62906]: https://github.com/vladromenko/explorer/commit/fd629068482570e18bc5d2d087133dc647f8a957
[cad-external]: https://github.com/dimwael/yahboom_m3pro_description/tree/bdac682e57a8eeaf7b18eece44bfbfc54de98e8a
[contract]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_policy_contract.py
[coordinator]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/room_cleanup.py
[dagger-external]: https://arxiv.org/abs/1011.0686
[diagnostic]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/live_room_diagnostic.py
[diffusion-external]: https://github.com/real-stanford/diffusion_policy
[early-learning]: https://github.com/vladromenko/explorer/blob/2d3aeae87d190f83449981fbdad5cf5e55673f36/docs/LEARNING-IMPLEMENTATION.ru.md
[entry]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/train_entry.py
[environment]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/environment.py
[eval-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EVALUATION_REPORT.md
[explorer-model]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/explorer_model.py
[guide]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING-GUIDE.ru.md
[implementation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/LEARNING-IMPLEMENTATION.ru.md
[intervention]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_intervention.py
[learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/learning.py
[lerobot-external]: https://huggingface.co/docs/lerobot/en/act
[lineage]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/episode_lineage.py
[local-grasp]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/local_grasp_experiment.py
[mjx-external]: https://mujoco.readthedocs.io/en/stable/mjx.html
[model-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/MODEL_AND_TRAINING.md
[neural]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/neural_recovery.py
[observer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/checkpoint_monitor.py
[onboard]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EXPLORER-ONBOARD-RELEASE-20261010.ru.md
[origin]: https://github.com/vladromenko/explorer
[phase]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/phase_learning.py
[r-adaptive17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/adaptive-learning-20261017.json
[r-audit12]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-audit-e7905c76974c-20261012.json
[r-auto16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-20261016.json
[r-causal17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/causal-repair-20261017.json
[r-correction16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-correction-20261016.json
[r-corrective15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-corrective-cycle-20261015.json
[r-hierarchy15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/hierarchical-rework-20261015.json
[r-jetson14]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-training-d6ebe13cae23-jetson-20261014.json
[r-macos]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-macos-20261011.json
[r-monitor17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-causal-monitor-20261017.json
[r-neural18]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/neural-recovery-20261018.json
[r-overnight]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-overnight-check-20261012.json
[record]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_demonstrations.py
[repair]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/failure_repair.py
[repo]: https://github.com/vladromenko/explorer-learning
[room-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/ROOM-CLEANUP.ru.md
[room-learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_learning.py
[room-policy]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_policy.py
[room-port]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_cleanup_port.py
[room08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/audits/20261008-room-resume.md
[runbook]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/RUNBOOK.md
[search]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/skill_training.py
[simopt-external]: https://arxiv.org/abs/1810.05687
[smolvla-external]: https://huggingface.co/docs/lerobot/en/smolvla
[stack]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/learning_stack.py
[transfer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_transfer.py
[unified]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/UNIFIED-AUTONOMY.ru.md
[validation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_validation.py
[vision]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/OBJECT-VISION.ru.md
## 15. Что исправлено в безопасности и критериях результата

Raw-прогноз с превышением допустимого шага сустава более 20° отклоняется. Выдаваемое изменение endpoint ограничено 2° на решение с integer/error-diffusion. Ошибочный cached chunk можно ограниченно перепланировать, но свежий непригодный прогноз не разрешается «лечить» бесконечным resample. MotionViolation выходит немедленно. Последний отказ захвата показывает работу защиты, а не успех политики. [Контракт][contract], [исполнение ACT][room-policy].

Требуются свежесть/идентичность RGB-D, новая цель после движения базы, достижимость камеры, реальные разные ракурсы, наблюдаемый hold, проверенный путь, размещение и отход. Истинные контакты/координаты нельзя подавать actor под видом сенсора. Консервативная CAD-проверка уточнялась только при доказанном геометрическом разделении, без уменьшения физического корпуса.

Историческое «когда-либо был успех» не должно означать текущую доставку: итог оценивает нынешнее нахождение внутри, отпускание и устойчивость после завершения действий. Нужны также наблюдаемое подтверждение и отсутствие столкновений. Отдельно запрещена подмена ученика teacher во время теста.

Старые названия артефактов вроде `validated_simulation` могут относиться к прежнему более слабому/локальному протоколу. Новая готовность полного room-сценария не наследуется из имени старого каталога. Протокол усилен с прежних двух групп по 10 до **двух по 50**, а не ослаблен для получения отчёта.


[act-external]: https://github.com/tonyzhaozh/act
[archive]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_archive.py
[audit-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING_AUDIT.md
[audit08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/CODE-AUDIT-20261008.ru.md
[bridge]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/lerobot_bridge.py
[c-0293b06]: https://github.com/vladromenko/explorer/commit/0293b068acf58542b8ec61605c75ab7feed8e802
[c-09cb728]: https://github.com/vladromenko/explorer/commit/09cb7282c2a9a52977329c6cff78470e9ebe1a3d
[c-0b231e4]: https://github.com/vladromenko/explorer/commit/0b231e48e2e7af1256d2a60a6cc67e6f1287556f
[c-0bd7728]: https://github.com/vladromenko/explorer/commit/0bd772860832ef54d8a65aed0d8abaa8f1bce0d7
[c-146a463]: https://github.com/vladromenko/explorer/commit/146a4630ef407efef09b5f6ec3944909b206634e
[c-1d6ddf5]: https://github.com/vladromenko/explorer/commit/1d6ddf538369e6c54b1fedb22f6e57ef2ec119d2
[c-1da8ecb]: https://github.com/vladromenko/explorer/commit/1da8ecb26efaa798d4293f126d7d123aff7985e5
[c-2161826]: https://github.com/vladromenko/explorer/commit/2161826271fb90efd9bd6e9ae7cc818590d09e51
[c-2d3aeae]: https://github.com/vladromenko/explorer/commit/2d3aeae87d190f83449981fbdad5cf5e55673f36
[c-2e484e3]: https://github.com/vladromenko/explorer/commit/2e484e39f98078d99b5e579c043bc1f6e5072ee6
[c-30271ba]: https://github.com/vladromenko/explorer/commit/30271bae7e0c53a3a06462fa55a7da962dace506
[c-3568b30]: https://github.com/vladromenko/explorer/commit/3568b300116926717f03b8b13cf01c3984702093
[c-37ae853]: https://github.com/vladromenko/explorer/commit/37ae853480c6dc88c7dbbebdfd348e38fa0876f2
[c-3fb6905]: https://github.com/vladromenko/explorer/commit/3fb69053bafc5185b70dcbef75988bf79d043bb2
[c-4146b1c]: https://github.com/vladromenko/explorer/commit/4146b1c6ba509b31431fcbde5daa291d184b869e
[c-42374fe]: https://github.com/vladromenko/explorer/commit/42374fe013127d9a3d82eb0a81011f51c29e05c8
[c-43304eb]: https://github.com/vladromenko/explorer/commit/43304eb10133002fe2fe11395d9f2d67317d7c16
[c-4386cc0]: https://github.com/vladromenko/explorer/commit/4386cc0cdf162de3dbaab6d3568335301146003d
[c-5b27e21]: https://github.com/vladromenko/explorer/commit/5b27e215b94ff0d483806391559f4dc83a8b7576
[c-5bb6914]: https://github.com/vladromenko/explorer/commit/5bb69148a733272d53f3897acbcab65a6a335f4b
[c-5db446d]: https://github.com/vladromenko/explorer/commit/5db446d32752698256ddb7e4a5c36cc258967605
[c-65c5738]: https://github.com/vladromenko/explorer/commit/65c57387e988998cc07349d5fc89e624dcb9d122
[c-67908f9]: https://github.com/vladromenko/explorer/commit/67908f956eea5b5f72db5c906bf989d8810c2504
[c-6d6394a]: https://github.com/vladromenko/explorer/commit/6d6394a4ba3a2e92381b2f45fbd60f4867e572b5
[c-726f8cb]: https://github.com/vladromenko/explorer/commit/726f8cbd72f7d4723d8a0e8500099efbae3dfb67
[c-853c4c0]: https://github.com/vladromenko/explorer/commit/853c4c0d9c418fac3a4b9f1318c9b036e3c11b33
[c-8c6ce11]: https://github.com/vladromenko/explorer/commit/8c6ce11c8f27c91e60aaa8c9623a55040a019706
[c-929497a]: https://github.com/vladromenko/explorer/commit/929497a7f85366983d2bfbdf4dea9910e727ced2
[c-934c479]: https://github.com/vladromenko/explorer/commit/934c47986e11621fe0e0a074e51866e2bc81b92b
[c-a5d9ea7]: https://github.com/vladromenko/explorer/commit/a5d9ea7c3f40324fe4c37a54bd3425b46bf34f65
[c-b4186e1]: https://github.com/vladromenko/explorer/commit/b4186e1749ad2ac12d845fc51e63adf74828fad8
[c-b6373e3]: https://github.com/vladromenko/explorer/commit/b6373e35bd827ba42f50d5d1b2e6954e575ed61e
[c-b828129]: https://github.com/vladromenko/explorer/commit/b8281297e4eb3f846bba22193710b04cacc6775a
[c-b9cd5e3]: https://github.com/vladromenko/explorer/commit/b9cd5e31f29d86935c60e24ff577878eaa785801
[c-bf36e4e]: https://github.com/vladromenko/explorer/commit/bf36e4ea370d44eb1757c093247bb1e10e167ff6
[c-c553017]: https://github.com/vladromenko/explorer/commit/c55301788dca39584bcf7d68f761580de2a6c7ac
[c-c746696]: https://github.com/vladromenko/explorer/commit/c7466967cbe00f26ff7ee867cd0ff8542d8b0050
[c-cfbd8eb]: https://github.com/vladromenko/explorer/commit/cfbd8eb0d2f6f0b4defa055aac9ea972380f6c84
[c-cfe4aa7]: https://github.com/vladromenko/explorer/commit/cfe4aa7044e75798384d8fd6269fcab84761820d
[c-d1c4d11]: https://github.com/vladromenko/explorer/commit/d1c4d1155fad093f5ca04c92343dd1c0d44a6fb5
[c-e9ad08b]: https://github.com/vladromenko/explorer/commit/e9ad08be2d3694369610d88676024015b21ec6d8
[c-f2ac13d]: https://github.com/vladromenko/explorer/commit/f2ac13dff1b378fc05600bd1443fb537441d289e
[c-fd62906]: https://github.com/vladromenko/explorer/commit/fd629068482570e18bc5d2d087133dc647f8a957
[cad-external]: https://github.com/dimwael/yahboom_m3pro_description/tree/bdac682e57a8eeaf7b18eece44bfbfc54de98e8a
[contract]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_policy_contract.py
[coordinator]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/room_cleanup.py
[dagger-external]: https://arxiv.org/abs/1011.0686
[diagnostic]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/live_room_diagnostic.py
[diffusion-external]: https://github.com/real-stanford/diffusion_policy
[early-learning]: https://github.com/vladromenko/explorer/blob/2d3aeae87d190f83449981fbdad5cf5e55673f36/docs/LEARNING-IMPLEMENTATION.ru.md
[entry]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/train_entry.py
[environment]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/environment.py
[eval-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EVALUATION_REPORT.md
[explorer-model]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/explorer_model.py
[guide]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING-GUIDE.ru.md
[implementation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/LEARNING-IMPLEMENTATION.ru.md
[intervention]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_intervention.py
[learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/learning.py
[lerobot-external]: https://huggingface.co/docs/lerobot/en/act
[lineage]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/episode_lineage.py
[local-grasp]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/local_grasp_experiment.py
[mjx-external]: https://mujoco.readthedocs.io/en/stable/mjx.html
[model-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/MODEL_AND_TRAINING.md
[neural]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/neural_recovery.py
[observer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/checkpoint_monitor.py
[onboard]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EXPLORER-ONBOARD-RELEASE-20261010.ru.md
[origin]: https://github.com/vladromenko/explorer
[phase]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/phase_learning.py
[r-adaptive17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/adaptive-learning-20261017.json
[r-audit12]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-audit-e7905c76974c-20261012.json
[r-auto16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-20261016.json
[r-causal17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/causal-repair-20261017.json
[r-correction16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-correction-20261016.json
[r-corrective15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-corrective-cycle-20261015.json
[r-hierarchy15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/hierarchical-rework-20261015.json
[r-jetson14]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-training-d6ebe13cae23-jetson-20261014.json
[r-macos]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-macos-20261011.json
[r-monitor17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-causal-monitor-20261017.json
[r-neural18]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/neural-recovery-20261018.json
[r-overnight]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-overnight-check-20261012.json
[record]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_demonstrations.py
[repair]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/failure_repair.py
[repo]: https://github.com/vladromenko/explorer-learning
[room-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/ROOM-CLEANUP.ru.md
[room-learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_learning.py
[room-policy]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_policy.py
[room-port]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_cleanup_port.py
[room08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/audits/20261008-room-resume.md
[runbook]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/RUNBOOK.md
[search]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/skill_training.py
[simopt-external]: https://arxiv.org/abs/1810.05687
[smolvla-external]: https://huggingface.co/docs/lerobot/en/smolvla
[stack]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/learning_stack.py
[transfer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_transfer.py
[unified]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/UNIFIED-AUTONOMY.ru.md
[validation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_validation.py
[vision]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/OBJECT-VISION.ru.md

## Teacher и runtime: разные роли и контракты

VERIFIED: геометрический teacher использует привилегированную геометрию симулятора для генерации данных. Вход actor ограничен RGB/state/clock. Скрытые диагностические поля не должны попадать в actor features; whitelist экспорта проверяется отдельно. Точные функции — `source/simulator/learning.py`, `teacher.py`, `room_teacher.py` и индекс функций.

Исторические426 успехов/700 teacher попыток относились к локальному сценарию. Они не доказывают устойчивый полный поиск и уборку. Новый full collector требует минимум8 полных успехов и4 успешных student-state corrections; наличие реализации и программных тестов не означает фактическое выполнение этой квоты.

В текущих записях максимальные target-state приращения по6 каналам примерно[1.217,2.4,2.4,3,3,3.6]°; runtime выдаёт не более2° за шаг. Это возможная причина накопленного отставания от демонстрации. Она не объясняет автоматически свежий скачок27.82° и не разрешает ослабить guard. Нужно сравнить teacher и runtime при одинаковых наблюдениях и исполнении.

Исторический cadence defect был конкретным: после10 ACT ticks выполнялись12 manual ticks, что давало22с физического времени против10с clock на200 решениях. Исправление cfe4aa7 прошло регрессию10/10с. Однако последующий прогон всё равно не поднял носок: исправление времени было необходимым компонентным ремонтом, а не решением навыка.

В нынешнем trace695 physical elapsed34.75с, actor clock6→40.75с. Поэтому старый дефект22/10 не объясняет этот отказ. Но смысл времени остаётся несогласованным: runtime находится вpick, тогда как учебные примеры на40.75с почти всегда относятся к отпусканию/отходу. Данные и вывод в05–06 отделяют этот факт от пока не доказанной причинной гипотезы.

Корректный teacher→student тест должен сравнивать не только команды, но фактическое следование суставов, видимость, контакт, удержание, отпускание и текущую устойчивость финального размещения. Ни увеличение train steps, ни совпадение array shapes не проверяют всё это.
