;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 Op-blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain DOOR)
  (:requirements :strips :typing)
  (:types handle block)
  (:predicates (rest ?x - handle)
           (notrest ?x - handle)
	       (open ?x - handle)
	       (closed ?x - handle)
	       (handempty)
	       (holding ?x - handle)
	       (inPath ?y - block)
	       (pathBlocked)
	       (pathClear)
	       )

  (:action grab
	     :parameters (?x - handle)
	     :precondition (and (rest ?x) (handempty) (pathClear))
	     :effect
	     (and (holding ?x) (rest ?x) (not (handempty)) (not (notrest ?x))))

  (:action push-down
	     :parameters (?x - handle)
	     :precondition (and (rest ?x) (holding ?x))
	     :effect
	     (and (not (rest ?x))
		   (holding ?x) (notrest ?x) (not (handempty))))

  (:action pull-open
	     :parameters (?x - handle)
	     :precondition (and (notrest ?x) (holding ?x) (closed ?x))
	     :effect
	     (and (holding ?x) (not (handempty))
		   (not (rest ?x)) (notrest ?x)
		   (open ?x) (not (closed ?x))))

  (:action release
	     :parameters (?x - handle)
	     :precondition (holding ?x)
	     :effect
	     (and (not (holding ?x)) (handempty)
		   (rest ?x) (not (notrest ?x))))

  (:action clear-path
         :parameters (?y - block)
         :precondition (and (inPath ?y) (pathBlocked))
	     :effect
	     (and (not (pathBlocked)) (pathClear) (not (inPath ?y))
		   )))

