
(define (problem blocks_operator_actionsOne3) (:domain blocks_operator_actions)
  (:objects
        blue - block
	green - block
	grey - block
	red - block
	yellow - block
  )
(:init
	(clear green)
	(clear grey)
	(clear red)
	(clear yellow)
	(handempty)
	(on grey blue)
	(ontable blue)
	(ontable green)
	(ontable red)
	(ontable yellow)
)
(:goal (and
	(on grey blue)
	(on blue green)
	(on yellow red)
	(clear yellow)
	(clear grey)))
)
  
