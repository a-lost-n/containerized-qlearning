build-image:
	docker build -t worker:latest worker/.m
	docker build -t master:latest master/.

build:
	kubectl apply -f worker/worker.yaml
	kubectl apply -f master/master.yaml
	kubectl apply -f master/master-service.yaml
	kubectl apply -f roles/role.yaml
	kubectl apply -f roles/roleAccount.yaml
	kubectl apply -f roles/roleBinding.yaml

pod-info:
	kubectl get pods

clean:
	kubectl delete --all deployment