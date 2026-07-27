
(define (problem blocks_operator_actionsZero1) (:domain blocks_operator_actions)
  (:objects
        blue - block
	green - block
	red - block
	yellow - block
  )
(:init
	(clear blue)
	(clear green)
	(clear red)
	(handempty)
	(on green yellow)
	(ontable blue)
	(ontable red)
	(ontable yellow)
)
(:goal (and
	(on blue green)
	(on green red)
	(on red yellow)))
)