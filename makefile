build-image:
	docker build -t freewer/worker:latest worker/.
	docker build -t freewer/master:latest master/.

build:
	kubectl apply -f worker/worker.yaml
	kubectl apply -f worker/worker-service.yaml
	kubectl apply -f master/master.yaml
	kubectl apply -f master/master-service.yaml
	kubectl apply -f roles/role.yaml
	kubectl apply -f roles/roleAccount.yaml
	kubectl apply -f roles/roleBinding.yaml

minikube:
	minikube start

stop:
	minikube stop

pod-info:
	kubectl get pods

clean:
	kubectl delete --all deployment