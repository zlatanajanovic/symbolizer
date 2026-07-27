
(define (problem blocks_operator_actionsZero1) (:domain blocks_operator_actions)
  (:objects
        blue - block
	green - block
	red - block
	yellow - block
  )
(:init
	(clear red)
	(handfull)
	(holding yellow)
	(on green blue)
	(on red green)
	(ontable blue)
)
(:goal (and
	(on blue green)
	(on green red)
	(on red yellow)))
)